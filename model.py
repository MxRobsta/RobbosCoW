import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerRegressorDual(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1, n_hearing=3):
        super().__init__()

        self.linear_in = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.hearing_embedding = nn.Embedding(n_hearing, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer_l = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_l = nn.TransformerEncoder(
            encoder_layer_l, num_layers=num_layers
        )

        encoder_layer_r = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_r = nn.TransformerEncoder(
            encoder_layer_r, num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model * 2, 1)

    def _preprocess(self, x, attention_mask, hearing_labels):
        x = self.linear_in(x)
        batch, _, _ = x.size()
        hearing_emb = self.hearing_embedding(hearing_labels).unsqueeze(1)
        cls_emb = self.cls_token.expand(batch, -1, -1)
        x[:, 0:1, :] = cls_emb
        x[:, 1:2, :] = hearing_emb
        x = self.pos_encoder(x)

        key_padding_mask = ~attention_mask
        key_padding_mask[:, 0] = False
        key_padding_mask[:, 1] = False
        return x, key_padding_mask

    def forward(self, feats_l, mask_l, feats_r, mask_r, hearing_labels):
        x_l, key_padding_mask_l = self._preprocess(feats_l, mask_l, hearing_labels)
        x_r, key_padding_mask_r = self._preprocess(feats_r, mask_r, hearing_labels)

        out_l = self.transformer_l(x_l, src_key_padding_mask=key_padding_mask_l)
        out_r = self.transformer_r(x_r, src_key_padding_mask=key_padding_mask_r)

        cls_l = out_l[:, 0, :]
        cls_r = out_r[:, 0, :]

        cls_concat = torch.cat([cls_l, cls_r], dim=1)
        pred = self.fc_out(cls_concat).squeeze(1)
        return pred
