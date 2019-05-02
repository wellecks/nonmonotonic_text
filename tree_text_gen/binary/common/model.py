import torch
import torch.nn as nn
import torch.nn.functional as F
from tree_text_gen.binary.common.attention import AttentionLayer

class LSTMDecoder(nn.Module):
    def __init__(self, config, tok2i, sampler, encoder):
        super(LSTMDecoder, self).__init__()
        self.fc_dim = config['fc_dim']
        self.dec_lstm_dim = config['dec_lstm_dim']
        self.dec_n_layers = config['dec_n_layers']
        self.n_classes = config['n_classes']
        self.word_emb_dim = config['word_emb_dim']
        self.device = config['device']
        self.longest_label = config['longest_label']
        self.model_type = config['model_type']
        self.aux_end = config.get('aux_end', False)
        self.encoder = encoder

        # -- Decoder
        self.dec_lstm_input_dim = config.get('dec_lstm_input_dim', self.word_emb_dim)
        self.dec_lstm = nn.LSTM(self.dec_lstm_input_dim, self.dec_lstm_dim, self.dec_n_layers, batch_first=True)
        self.dec_emb = nn.Embedding(self.n_classes, self.word_emb_dim)
        if config['nograd_emb']:
            self.dec_emb.weight.requires_grad = False
        self.dropout = nn.Dropout(p=config['dropout'])

        # Layers for mapping LSTM output to scores
        self.o2emb = nn.Linear(self.dec_lstm_dim, self.word_emb_dim)
        # Optionally use the (|V| x d_emb) matrix from the embedding layer here.
        if config['share_inout_emb']:
            self.out_bias = nn.Parameter(torch.zeros(self.n_classes).uniform_(0.01))
            self.emb2score = lambda x: F.linear(x, self.dec_emb.weight, self.out_bias)
        else:
            self.emb2score = nn.Linear(self.word_emb_dim, self.n_classes)

        self.register_buffer('START', torch.LongTensor([tok2i['<s>']]))
        self.sampler = sampler
        self.end = tok2i['<end>']

        if self.aux_end:
            self.o2stop = nn.Sequential(nn.Linear(self.dec_lstm_dim, self.word_emb_dim),
                                        nn.ReLU(),
                                        self.dropout,
                                        nn.Linear(self.word_emb_dim, 1),
                                        nn.Sigmoid())

        if self.model_type == 'translation':
            self.enc_to_h0 = nn.Linear(config['enc_lstm_dim'] * config['num_dir_enc'],
                                       self.dec_n_layers * self.dec_lstm_dim)
            self.attention = AttentionLayer(input_dim=self.dec_lstm_dim,
                                            hidden_size=self.dec_lstm_dim,
                                            bidirectional=config['num_dir_enc'] == 2)
            self.decode = self.forward_decode_attention
            self.dec_emb.weight = self.encoder.emb.weight
        else:
            self.decode = self.forward_decode

    def o2score(self, x):
        x = self.o2emb(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.emb2score(x)
        return x

    def forward(self, xs=None, oracle=None, max_steps=None, num_samples=None, return_p_oracle=False):
        B = num_samples if num_samples is not None else xs.size(0)
        encoder_output = self.encode(xs)
        hidden = self.init_hidden(encoder_output if encoder_output is not None else B)
        scores = []
        samples = []
        p_oracle = []
        self.sampler.reset(bsz=B)
        if self.training:
            done = oracle.done()
            xt = self.START.detach().expand(B, 1)
            t = 0
            while not done:
                hidden = self.process_hidden_pre(hidden, xt, encoder_output)
                score_t, _, hidden = self.decode(xt, hidden, encoder_output)
                xt = self.sampler(score_t, oracle, training=True)
                hidden = self.process_hidden_post(hidden, xt, encoder_output)
                p_oracle.append(oracle.distribution())
                oracle.update(xt)
                samples.append(xt)
                scores.append(score_t)
                t += 1
                done = oracle.done()
                if max_steps and t == max_steps:
                    done = True

            self.longest_label = max(self.longest_label, t)
        else:
            with torch.no_grad():
                xt = self.START.detach().expand(B, 1)
                for t in range(self.longest_label):
                    hidden = self.process_hidden_pre(hidden, xt, encoder_output)
                    score_t, _, hidden = self.decode(xt, hidden, encoder_output)
                    xt = self.sampler(score_t, oracle=None, training=False)
                    hidden = self.process_hidden_post(hidden, xt, encoder_output)
                    scores.append(score_t)
                    samples.append(xt)

        samples = torch.cat(samples, 1)
        if not self.aux_end:
            scores = torch.cat(scores, 1)
        if return_p_oracle:
            p_oracle = torch.stack(p_oracle, 1)
            return scores, samples, p_oracle
        return scores, samples

    def encode(self, xs):
        if self.model_type == 'unconditional':
            encoder_output = None
        elif self.model_type == 'bagorder':
            encoder_output = self.encoder(xs)
            encoder_output = encoder_output.unsqueeze(0).expand(self.dec_n_layers, xs.size(0),
                                                                self.dec_lstm_dim).contiguous()
        elif self.model_type == 'translation':
            encoder_output = self.encoder(xs)
        else:
            raise NotImplementedError('Unsupported model type %s' % self.model_type)
        return encoder_output

    def forward_decode(self, xt, hidden, encoder_output):
        xes = self.embed_input(xt)
        xes = self.dropout(xes)
        lstm_output, hidden = self.dec_lstm(xes, hidden)
        scores = self.o2score(lstm_output)
        if self.aux_end:
            stop = self.o2stop(lstm_output).squeeze(2)
            scores = (scores, stop)
        return scores, lstm_output, hidden

    def forward_decode_attention(self, xt, hidden, encoder_output):
        enc_states, enc_hidden, attn_mask = encoder_output
        xes = self.embed_input(xt)
        xes = self.dropout(xes)
        lstm_output, hidden = self.dec_lstm(xes, hidden)
        lstm_output, _ = self.attention(lstm_output, hidden, (enc_states, attn_mask))
        scores = self.o2score(lstm_output)
        if self.aux_end:
            stop = self.o2stop(lstm_output).squeeze(2)
            scores = (scores, stop)
        return scores, lstm_output, hidden

    def embed_input(self, xt):
        return self.dec_emb(xt)

    def init_hidden(self, encoder_output):
        N = self.dec_n_layers
        D = self.dec_lstm_dim
        if self.model_type == 'unconditional':
            B = encoder_output
            hidden = (torch.zeros(N, B, D, device=self.device),
                      torch.zeros(N, B, D, device=self.device))
        elif self.model_type == 'bagorder':
            B = encoder_output.size(1)
            hidden = (encoder_output,
                      torch.zeros(N, B, D, device=self.device))
        elif self.model_type == 'translation':
            _, last_hidden, _ = encoder_output
            B = last_hidden.size(0)
            hidden = (self.enc_to_h0(last_hidden).view(B, N, D).transpose(0, 1),
                      torch.zeros(N, B, D, device=self.device))
        else:
            raise NotImplementedError('Unsupported model type %s' % self.model_type)
        return hidden

    def process_hidden_pre(self, hidden, input_token, encoder_output):
        return hidden

    def process_hidden_post(self, hidden, sampled_token, encoder_output):
        # Add the encoder embedding for bagorder task
        if self.model_type == "bagorder":
            hidden = (hidden[0] + encoder_output, hidden[1])
        return hidden

