import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class STGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, num_nodes=13):
        super().__init__()
        self.num_nodes = num_nodes
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lstm = nn.LSTM(hidden_channels * num_nodes, latent_dim, batch_first=True)
        self.latent_dim = latent_dim

    def forward(self, x, edge_index, batch):
        # x: (total_nodes, in_channels)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # reshape to (batch_size, seq_len, num_nodes * hidden_channels)
        x = x.view(-1, self.num_nodes * x.size(-1))
        # предполагаем, что seq_len фиксирован
        x = x.view(batch.max().item()+1, -1, self.num_nodes * x.size(-1))
        _, (h, _) = self.lstm(x)
        return h[-1]  # (batch, latent_dim)

class STGCNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, out_channels, seq_len, num_nodes=13):
        super().__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.lstm = nn.LSTM(latent_dim, hidden_channels * num_nodes, batch_first=True)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, z, edge_index, batch_size):
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z)
        out = out.reshape(batch_size * self.seq_len * self.num_nodes, -1)
        out = self.fc(out)
        return out

class STGCNAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, seq_len, num_nodes=13):
        super().__init__()
        self.encoder = STGCNEncoder(in_channels, hidden_channels, latent_dim, num_nodes)
        self.decoder = STGCNDecoder(latent_dim, hidden_channels, in_channels, seq_len, num_nodes)

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.batch)
        recon = self.decoder(z, data.edge_index, data.batch.max().item()+1)
        return recon, z