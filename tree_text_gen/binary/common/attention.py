import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """Based on the ParlAI implementation, with only global, post, concat attention for simplicity.
    Ref: (https://github.com/facebookresearch/ParlAI/blob/
          0f79ebd380b6d59e7b2330b1a0e75d1f8a1b4884/parlai/agents/seq2seq/modules.py#L459)
    """
    def __init__(self, input_dim, hidden_size, bidirectional):
        super().__init__()
        hXdirs = input_dim * (2 if bidirectional else 1)
        self.attn_combine = nn.Linear(hXdirs + input_dim, hidden_size, bias=False)
        self.attn = nn.Linear(hidden_size + hXdirs, hidden_size, bias=False)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input, hidden, attn_params):
        if type(hidden) == tuple:
            hidden = hidden[0]
        last_hidden = hidden[-1]
        enc_states, attn_mask = attn_params
        B, T, hszXnumdir = enc_states.size()
        numlayersXnumdir = last_hidden.size(1)

        # Compute attention scores using concatenation of decoder hidden state and encoder states.
        hid = last_hidden.unsqueeze(1)
        hid = hid.expand(B, T, numlayersXnumdir)
        h_merged = torch.cat((enc_states, hid), 2)
        active = torch.tanh(self.attn(h_merged))
        attn_w_premask = self.attn_v(active).squeeze(2)

        # Normalize the attention scores, after applying mask.
        if attn_mask is not None:
            attn_w_premask.masked_fill_((1 - attn_mask), -1e20)
        attn_weights = F.softmax(attn_w_premask, dim=1)

        # Apply the attention (weighted sum of `enc_states`), combine the result with input.
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_states)
        merged = torch.cat((input.squeeze(1), attn_applied.squeeze(1)), 1)
        output = torch.tanh(self.attn_combine(merged).unsqueeze(1))
        return output, attn_weights