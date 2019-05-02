import torch
import torch.nn as nn

class BOWEncoder(nn.Module):
    def __init__(self, config, tok2i):
        super(BOWEncoder, self).__init__()
        self.word_emb_dim = config['word_emb_dim']
        self.device = config['device']
        self.output_size = config['dec_lstm_dim']
        self.enc_emb = nn.Embedding(len(tok2i), self.word_emb_dim, padding_idx=tok2i['<p>'])
        self.emb2out = nn.Linear(self.word_emb_dim, self.output_size)
        self.pad = tok2i['<p>']
        if config['nograd_emb']:
            self.enc_emb.weight.requires_grad = False

    def forward(self, xs):
        xes = self.enc_emb(xs)
        xes = self.emb2out(xes.view(-1, self.word_emb_dim))
        xes = xes.view(xs.size(0), xs.size(1), -1)
        enc = xes.sum(1) / torch.clamp(xs.ne(self.pad).sum(1, keepdim=True).float(), min=1.0)
        return enc

class LSTMEncoder(torch.nn.Module):
    def __init__(self, config, tok2i):
        super(LSTMEncoder, self).__init__()
        self.num_dir_enc = config['num_dir_enc']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.rnn = nn.LSTM(config['word_emb_dim'], config['enc_lstm_dim'], config['enc_n_layers'],
                           batch_first=True, bidirectional=config['num_dir_enc'] == 2, dropout=config['dropout'])
        self.emb = nn.Embedding(config['n_classes'], config['word_emb_dim'], padding_idx=tok2i['<p>'])
        self.dropout = nn.Dropout(p=config['dropout'])
        self.pad = tok2i['<p>']

    def forward(self, input):
        xs, xs_lens = input
        xes = self.emb(xs)
        xes = self.dropout(xes)
        attn_mask = xs.ne(self.pad)
        pack = torch.nn.utils.rnn.pack_padded_sequence(xes, xs_lens, batch_first=True)
        encoder_output, (h, c) = self.rnn(pack)
        encoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                               encoder_output, batch_first=True, total_length=xes.size(1))
        h = h[-self.num_dir_enc:,:,:]
        h = h.transpose(0, 1).contiguous().view(xs.size(0), self.num_dir_enc * self.enc_lstm_dim)
        return encoder_output, h, attn_mask

