import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, GlobalAttention


class ResidualCGBlock(nn.Module):
    """
    Residual CGConv block with normalization, activation, and dropout.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = CGConv(channels=hidden_dim, dim=edge_dim, aggr="add")
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        identity = x
        out = self.conv(x, edge_index, edge_attr)
        out = self.norm(out)
        out = F.silu(out)   # smoother than ReLU
        out = self.dropout(out)
        return identity + out


class CGCNNRegressorStrong(nn.Module):
    """
    Stronger CGCNN-style regressor with:
    - embedding + input projection
    - residual CGConv blocks
    - batch normalization
    - gated global pooling
    - deeper prediction head

    This architecture is usually more stable and generalises better than
    a plain stack of CGConv + mean pooling.
    """

    def __init__(
        self,
        num_embeddings: int = 100,
        atom_emb_dim: int = 128,
        edge_dim: int = 50,
        hidden_dim: int = 128,
        num_conv_layers: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Atom embedding
        self.embedding = nn.Embedding(num_embeddings, atom_emb_dim)

        # Project embeddings to hidden dimension used by message passing
        self.input_proj = nn.Sequential(
            nn.Linear(atom_emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Residual CGConv blocks
        self.convs = nn.ModuleList([
            ResidualCGBlock(hidden_dim=hidden_dim, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_conv_layers)
        ])

        # Gated pooling learns which atoms matter more
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.pool = GlobalAttention(gate_nn=self.pool_gate)

        # Stronger regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        # data.x expected to contain atomic numbers / atom indices
        x = data.x.view(-1).long()
        x = self.embedding(x)
        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)

        x = self.pool(x, data.batch)
        x = self.head(x)
        return x.view(-1)