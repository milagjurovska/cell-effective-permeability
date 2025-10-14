import os
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

class DescAwareRegressor(nn.Module):
    predictor_name = "desc_aware_regressor"

    def __init__(self, pred_input_channels: int, desc_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(pred_input_channels + desc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_emb: torch.Tensor, batch: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        g = global_mean_pool(node_emb, batch)
        batch_size = g.size(0)
        desc_dim = desc.size(0) // batch_size if desc.dim() == 1 else desc.size(-1)
        desc = desc.view(batch_size, desc_dim)
        z = torch.cat([g, desc], dim=-1)
        z = self.dropout(z)
        return self.mlp(z).squeeze(-1)

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(folder, "predictor.pth"))

    @classmethod
    def from_config(cls, cfg):
        return cls(
            pred_input_channels=getattr(cfg, "pred_input_channels"),
            desc_dim=getattr(cfg, "desc_dim"),
            hidden_dim=getattr(cfg, "hidden_dim", 128),
            dropout=getattr(cfg, "dropout", 0.2),
        )
