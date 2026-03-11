import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool


class CGCNNRegressor(nn.Module):
    """
    A simple CGCNN-style regressor for crystal property prediction.
    """

    def __init__(
        self,
        num_embeddings: int = 100,
        atom_emb_dim: int = 64,
        edge_dim: int = 50,
        hidden_dim: int = 128,
        num_conv_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, atom_emb_dim)

        self.convs = nn.ModuleList()
        in_dim = atom_emb_dim
        for _ in range(num_conv_layers):
            self.convs.append(CGConv(channels=in_dim, dim=edge_dim, aggr="add"))

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = data.x.view(-1)
        x = self.embedding(x)

        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
            x = F.softplus(x)

        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x.view(-1)