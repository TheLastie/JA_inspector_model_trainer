import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class TransformerAE(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, latent_dim=64, seq_len=30):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.latent_fc = nn.Linear(d_model, latent_dim)

        self.latent_to_dmodel = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        memory = self.encoder(x)
        latent = self.latent_fc(memory.mean(dim=1))  # пулинг по времени
        # декодирование
        latent = self.latent_to_dmodel(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        out = self.decoder(latent, memory)
        out = self.output_proj(out)
        return out, latent