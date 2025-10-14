import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATStack(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.2, attn_dropout: float = 0.0):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=attn_dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=attn_dropout)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.graph_output_channels = hidden_dim * heads

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x, edge_index)
        x = self.act(x)
        x = self.gat2(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        return x

    def save(self, folder: str):
        import os, torch
        os.makedirs(folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(folder, 'model_head.pth'))
