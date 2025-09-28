import torch
from torch import nn
from torch_geometric.nn import GINConv


def mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden)
    )


class GINStack(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2, train_eps: bool = True):
        super().__init__()
        self.conv1 = GINConv(mlp(in_dim, hidden_dim), train_eps=train_eps)
        self.conv2 = GINConv(mlp(hidden_dim, hidden_dim), train_eps=train_eps)
        self.dropout = nn.Dropout(dropout)
        self.graph_output_channels = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        return x

    def save(self, folder: str):
        import os, torch
        os.makedirs(folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(folder, 'model_head.pth'))
