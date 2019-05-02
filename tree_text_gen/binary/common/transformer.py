""" Transformer implementation based on a few sources:
    - http://nlp.seas.harvard.edu/2018/04/03/attention.html
    - https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py
    - https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/decoders/transformer.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tree_text_gen.binary.common.util as util


class SmallConfig(object):
    def __init__(self, device):
        self.d_model = 256
        self.d_ff = 1024
        self.num_attention_heads = 4
        self.dropout = 0.1
        self.num_layers = 4
        self.device = device
        self.share_embeddings = False
        self.auxiliary_end = True
        self.tree_encoding = True
        self.longest_label = 50  # updated during training


class Transformer(nn.Module):
    def __init__(self, config, src_vocab, trg_vocab, loss_fn):
        super(Transformer, self).__init__()
        self.d_model = config.d_model
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.emb_src = nn.Sequential(Embeddings(config.d_model, len(src_vocab)),
                                     PositionalEncoding(config.d_model, config.dropout, config.device))
        self.emb_trg = nn.Sequential(Embeddings(config.d_model, len(trg_vocab)),
                                     PositionalEncoding(config.d_model, config.dropout, config.device))

        self.pad_idx = trg_vocab['<p>']
        self.end_idx = trg_vocab['<end>']
        self.n_classes = len(trg_vocab)
        self.o2score = nn.Linear(config.d_model, self.n_classes)
        self.device = config.device
        self.longest_label = config.longest_label
        self.register_buffer('START', torch.LongTensor([trg_vocab['<s>']]))

        self.tree_encoding = config.tree_encoding
        if config.tree_encoding:
            self.tree_emb = TreeEncoding(config.d_model, config.device, trg_vocab['<end>'])

        self.auxiliary_end = config.auxiliary_end
        if config.auxiliary_end:
            self.o2stop = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                        nn.ReLU(),
                                        nn.Dropout(config.dropout),
                                        nn.Linear(self.d_model, 1),
                                        nn.Sigmoid())
        if config.share_embeddings:
            self.emb_src[0].lut.weight = self.emb_trg[0].lut.weight
            self.o2score.weight = self.emb_trg[0].lut.weight

        self.init_parameters()
        self.loss_fn = loss_fn
        self.stop_prob = 0.5

    def encode(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(-2)
        src_emb = self.emb_src(src)
        out = self.encoder(src_emb, src_mask)
        return out, src_mask

    def decode(self, memory, src_mask, trg, **kwargs):
        if trg is not None:
            trg_mask = (trg != self.pad_idx).unsqueeze(-2)
            sub_mask = self.subsequent_mask(trg.size(-1), trg_mask.device)
            trg_mask = trg_mask & sub_mask
        else:
            trg_mask = None

        trg_emb = self.emb_trg(trg)

        if self.tree_encoding:
            if 'tree_emb' in kwargs:  # for incremental decoding
                tree_emb, tree_emb_pre, parents = self.tree_emb.step(kwargs['tree_emb'], trg, kwargs['parents'], kwargs['t'])
                trg_emb = trg_emb + tree_emb
                out = self.decoder(trg_emb, memory, src_mask, trg_mask)
                return out, tree_emb_pre, parents
            else:
                tree_emb = self.tree_emb(trg)
                trg_emb = trg_emb + tree_emb

        out = self.decoder(trg_emb, memory, src_mask, trg_mask)
        return out

    def forward(self, xs, ys, p_oracle, oracle_cls=None, **kwargs):
        # Training case; sampling a mixed trajectory involving the oracle.
        if oracle_cls is not None:
            oracle = oracle_cls(ys.detach(), self.n_classes, self.tok2i, self.i2tok, **kwargs['oracle_flags'])
            scores, p_oracle, ys = self.forward_with_oracle(xs, oracle, **kwargs)
            loss = self.loss_fn(scores, ys, p_oracle, self.end_idx, **kwargs['loss_flags'])
            return loss

        # Training case; obtaining scores for a trajectory.
        elif ys is not None:
            scores = self.forward_scores(xs, ys, **kwargs)
            loss = self.loss_fn(scores, ys, p_oracle, self.end_idx, **kwargs['loss_flags'])
            return loss

        # Inference case
        else:
            ys = self.forward_inference(xs, **kwargs)
            return [], ys

    def forward_scores(self, xs, ys, **kwargs):
        xs = xs[0]
        enc, src_mask = self.encode(xs)
        y0 = self.START.detach().expand(xs.size(0), 1)
        ys = torch.cat((y0, ys), 1)
        dec = self.decode(enc, src_mask, ys)
        scores = self.o2score(dec)[:, :-1, :]
        if self.auxiliary_end:
            stops = self.o2stop(dec)[:, :-1, 0]
            scores = (scores, stops)
        return scores

    def forward_inference(self, xs, **kwargs):
        with torch.no_grad():
            xs = xs[0]
            enc, src_mask = self.encode(xs)
            yt = self.START.detach().expand(xs.size(0), 1)
            ys = yt
            B = xs.size(0)
            if self.tree_encoding:
                tree_emb = torch.zeros(B, 1, self.d_model, device=xs.device)
                parents = [[] for _ in range(B)]

            for t in range(self.longest_label):
                if self.tree_encoding:
                    dec, tree_emb, parents = self.decode(enc, src_mask, ys, tree_emb=tree_emb, parents=parents, t=t)
                else:
                    dec = self.decode(enc, src_mask, ys)

                yt = self.generate(dec[:, -1:, :])
                ys = torch.cat((ys, yt), 1)
            samples = ys[:, 1:]  # don't include the start symbol
            return samples

    def forward_with_oracle(self, xs, oracle, **kwargs):
        sampler = MixedPolicyCorrectSampler(GreedySampler(self.end_idx), self.end_idx, kwargs['beta'])
        xs = xs[0]
        enc, src_mask = self.encode(xs)
        yt = self.START.detach().expand(xs.size(0), 1)
        ys = yt
        p_oracle = []
        B = xs.size(0)

        if self.tree_encoding:
            tree_emb = torch.zeros(B, 1, self.d_model, device=xs.device)
            parents = [[] for _ in range(B)]

        for t in range(kwargs['max_steps']):
            if self.tree_encoding:
                dec, tree_emb, parents = self.decode(enc, src_mask, ys, tree_emb=tree_emb, parents=parents, t=t)
            else:
                dec = self.decode(enc, src_mask, ys)
            p_oracle_t = oracle.distribution()
            scores = self.o2score(dec)
            if self.auxiliary_end:
                stops = self.o2stop(dec)
                yt = sampler((scores[:, -1:, :], stops[:, -1:, :]), p_oracle_t, aux_stop=self.auxiliary_end)
            else:
                yt = sampler(scores[:, -1:, :], p_oracle_t, aux_stop=self.auxiliary_end)

            ys = torch.cat((ys, yt), 1)

            oracle.update(yt)
            p_oracle.append(p_oracle_t)

        p_oracle = torch.stack(p_oracle, 1)
        if self.auxiliary_end:
            scores = (scores, stops.squeeze(2))
        samples = ys[:, 1:]  # don't include the start symbol
        return scores, p_oracle, samples

    def generate(self, x):
        log_ps = F.log_softmax(self.o2score(x), dim=-1)
        yt = log_ps.argmax(2)
        if self.auxiliary_end:
            s = (self.o2stop(x) >= self.stop_prob).view(-1)
            yt = yt.masked_fill(s.unsqueeze(1), self.end_idx)
        return yt

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def subsequent_mask(self, size, device):
        subsequent_mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.uint8), 1).unsqueeze(0)
        subsequent_mask = subsequent_mask == 0
        return subsequent_mask


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
        self.norm = LayerNorm(config.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = LayerNorm(config.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(config.num_attention_heads,
                                              config.d_model,
                                              config.dropout)
        self.src_attn = MultiHeadedAttention(config.num_attention_heads,
                                             config.d_model,
                                             config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model,
                                                    config.d_ff,
                                                    config.dropout)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.norm3 = LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.src_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        out = self.lut(x)*math.sqrt(self.d_model)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.w_1(x)))
        out = self.w_2(x)
        return out


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return out


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(config.num_attention_heads,
                                              config.d_model,
                                              config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model,
                                                    config.d_ff,
                                                    config.dropout)
        self.d_model = config.d_model
        self.norm1 = LayerNorm(self.d_model)
        self.norm2 = LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        # NOTE: the exact use of layernorm varies by implementation..
        x0 = x
        x = self.self_attn(x, x, x, mask)
        x = self.norm1(x0 + self.dropout(x))

        x0 = x
        x = self.feed_forward(x)
        x = self.norm2(x0 + self.dropout(x))
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_attention_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = d_model // num_attention_heads
        self.all_head_size = num_attention_heads*self.attention_head_size

        self.query = nn.Linear(d_model, self.all_head_size)
        self.key = nn.Linear(d_model, self.all_head_size)
        self.value = nn.Linear(d_model, self.all_head_size)

        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def transpose_for_scores(self, x):
        B, T, D = x.size()
        new_x_shape = (B, T, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = self.transpose_for_scores(query)  # => B x num_heads x T x head_size
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e20)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        weighted_values = torch.matmul(attention_probs, value)
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous()
        new_shape = weighted_values.size()[:-2] + (self.all_head_size,)
        weighted_values = weighted_values.view(*new_shape)
        return weighted_values


class TreeEncoding(nn.Module):
    def __init__(self, d_model, device, end_idx):
        super(TreeEncoding, self).__init__()
        self.p = nn.Parameter(torch.ones(1))
        self.d_model = d_model
        self.end_idx = end_idx
        self.device = device

        zeros = torch.zeros(d_model, dtype=torch.float)
        self.register_buffer('zeros', zeros)

        left = torch.tensor([1, 0], dtype=torch.float)
        self.register_buffer('left', left)
        right = torch.tensor([0, 1], dtype=torch.float)
        self.register_buffer('right', right)

        range = torch.arange(d_model, dtype=torch.float)
        self.register_buffer('range', range)

        self.LEFT = 0
        self.RIGHT = 1

    def step(self, X, tokens, parents, t):
        # For incremental decoding
        X = X.detach()
        if t > 0:
            Xt = torch.zeros(X.size(0), self.d_model, device=X.device)
        B = X.size(0)
        for i in range(B):
            if t == 0:
                x_it = self.zeros.detach()
                # Enqueue parent index and whether the dequeue-er will be a left or right child.
                parents[i].append((0, self.LEFT))
                parents[i].append((0, self.RIGHT))
            else:
                # Get parent index and whether this is a left or right child.
                parent, lr = parents[i][0]
                x_parent = X[i, parent]

                if lr == self.LEFT:
                    x_it = torch.cat((self.left.detach(), x_parent[:-2]))
                else:
                    x_it = torch.cat((self.right.detach(), x_parent[:-2]))

                if tokens[i, t] != self.end_idx:
                    parents[i].append((t, self.LEFT))
                    parents[i].append((t, self.RIGHT))

                # Dequeue (the if may eval to False after generation has ended, which is fine)
                if len(parents[i]) > 1:
                    parents[i] = parents[i][1:]

            if t > 0:
                Xt[i] = x_it.detach()

        X_pre = X
        if t > 0:
            poly = torch.pow(self.p, self.range.detach()).unsqueeze(0).unsqueeze(0)
            X_pre = torch.cat((X, Xt.unsqueeze(1)), 1)
            X = X_pre*poly

        return X, X_pre, parents

    def forward(self, tokens):
        # For non-incremental decoding
        B, T = tokens.size()
        parents = [[] for _ in range(B)]
        X = torch.zeros(B, T, self.d_model).to(tokens.device)
        with torch.no_grad():
            for i in range(B):
                for t in range(T):
                    if t == 0:
                        x_it = self.zeros.detach()
                        # Enqueue parent index and whether the dequeue-er will be a left or right child.
                        parents[i].append((0, self.LEFT))
                        parents[i].append((0, self.RIGHT))
                    else:
                        # Get parent index and whether this is a left or right child.
                        parent, lr = parents[i][0]
                        x_parent = X[i, parent]

                        if lr == self.LEFT:
                            x_it = torch.cat((self.left.detach(), x_parent[:-2]))
                        else:
                            x_it = torch.cat((self.right.detach(), x_parent[:-2]))

                        if tokens[i, t] != self.end_idx:
                            parents[i].append((t, self.LEFT))
                            parents[i].append((t, self.RIGHT))

                        # Dequeue (the if may eval to False after generation has ended, which is fine)
                        if len(parents[i]) > 1:
                            parents[i] = parents[i][1:]

                    X[i, t] = x_it.detach()

        poly = torch.pow(self.p, self.range.detach()).unsqueeze(0).unsqueeze(0)
        X = X.cuda()*poly
        return X


class MixedPolicyCorrectSampler(object):
    """A training sampler that samples from a distribution proportional to the policy's
       distribution restricted to correct actions."""
    def __init__(self, base_sampler, end_idx, beta):
        self.base_sampler = base_sampler
        self.end_idx = end_idx
        self.beta = beta

    def __call__(self, scores, p_oracle, aux_stop=True, **kwargs):
        with torch.no_grad():
            use_oracle = np.random.binomial(1, p=self.beta) > 0
            if use_oracle:
                samples = p_oracle.multinomial(1)
            else:
                correct_actions_mask = p_oracle.gt(0).unsqueeze(1).float()

                if aux_stop:
                    token_scores, stop_probs = scores
                    # Prevent invalid stop (use oracle stop)
                    stop_probs = correct_actions_mask[:, :, self.end_idx]
                else:
                    token_scores = scores

                # this was necessary for preventing inf...
                token_scores = torch.clamp(token_scores, -40, 40)
                ps = util.masked_softmax(token_scores, correct_actions_mask, dim=2)

                if aux_stop:
                    samples = self.base_sampler((ps, stop_probs))
                else:
                    samples = self.base_sampler(ps, aux_stop=aux_stop)
        return samples


class GreedySampler(object):
    def __init__(self, end_idx):
        self.end_idx = end_idx

    def __call__(self, scores, aux_stop=True, **kwargs):
        if aux_stop:
            scores, stop_probs = scores

        tokens = scores.argmax(2)

        if aux_stop:
            stop = (stop_probs >= 0.5).squeeze(1)
            # If stop was predicted, put the <end> symbol instead of the token
            tokens[stop, :] = self.end_idx
        return tokens