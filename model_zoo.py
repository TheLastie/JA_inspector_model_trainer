"""
Модуль model_zoo.py – коллекция нейросетевых моделей для анализа упражнений.

Все генеративные модели (RAE, TransformerAE, STGCN_AE, VAE) возвращают
восстановленную последовательность и обучаются с MSE Loss, что даёт
значения ошибки в диапазоне ~0.001–0.05 после обучения на нормализованных данных.

Analyzer – модель классификации ошибок по суставам, метрика Accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, List


# ====================== Базовые компоненты ======================

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ====================== RAE (Recurrent Autoencoder) ======================

class RecurrentAutoencoder(nn.Module):
    """LSTM-автоэнкодер с авторегрессионным декодером."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        autoregressive: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.autoregressive = autoregressive

        # Энкодер
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Декодер
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

        if self.autoregressive:
            self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.out_to_hidden = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Encode
        _, (h, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h[-1])  # (batch, latent_dim)

        # Decode
        decoder_hidden = self.decoder_fc(latent)
        h_t = decoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_t = torch.zeros_like(h_t)

        if self.autoregressive:
            decoder_input = self.start_token.repeat(batch_size, 1, 1)
            outputs = []
            for t in range(seq_len):
                out, (h_t, c_t) = self.decoder_lstm(decoder_input, (h_t, c_t))
                pred = self.decoder_out(out)
                outputs.append(pred)
                decoder_input = self.out_to_hidden(pred)
            recon = torch.cat(outputs, dim=1)
        else:
            decoder_input = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            out, _ = self.decoder_lstm(decoder_input, (h_t, c_t))
            recon = self.decoder_out(out)

        return recon, latent


# ====================== Transformer Autoencoder ======================

class TransformerAutoencoder(nn.Module):
    """Трансформерный автоэнкодер."""
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        latent_dim: int = 64,
        seq_len: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.latent_fc = nn.Linear(d_model, latent_dim)
        self.latent_to_dmodel = nn.Linear(latent_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Encode
        x_proj = self.input_proj(x)
        x_pe = self.pos_encoder(x_proj)
        memory = self.encoder(x_pe)
        latent = self.latent_fc(memory.mean(dim=1))  # pooling

        # Decode
        decoder_input = self.latent_to_dmodel(latent).unsqueeze(1).repeat(1, seq_len, 1)
        out = self.decoder(decoder_input, memory)
        recon = self.output_proj(out)

        return recon, latent


# ====================== VAE (LSTM-VAE) ======================

class LSTMVAE(nn.Module):
    """Вариационный автоэнкодер с LSTM."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.encoder_lstm(x)
        h_last = h[-1]
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        z = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(z)
        return self.decoder_out(out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar


# ====================== Analyzer (детектор ошибок) ======================

class ExerciseAnalyzer(nn.Module):
    """Классификатор ошибок по суставам на основе предобученного энкодера."""
    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_joints: int = 11,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.encoder = encoder
        self.num_joints = num_joints

        self.shared_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Бинарные классификаторы для каждого сустава
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 2)  # 0 - норма, 1 - ошибка
            ) for _ in range(num_joints)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        latent = self.encoder(x)
        shared = self.shared_fc(latent)
        return [clf(shared) for clf in self.classifiers]  # список тензоров (batch, 2)