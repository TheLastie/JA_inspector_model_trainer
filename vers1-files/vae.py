import torch.nn as nn
import torch.nn.functional as F

class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, num_layers=2):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h, _) = self.encoder_lstm(x)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        z = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(z)
        return self.output_fc(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar