import json
import os
import torch
from torch import nn

from gat_model.configs.base_config import BaseConfig
from gat_model.configs.model_config import ModelConfig
from gat_model.models.gat_stack import GATStack
from gat_model.models.predictor import Predictor


class Model(nn.Module):
    def __init__(self, model_head: nn.Module, prediction_head: nn.Module, freeze_head: bool = False) -> None:
        super().__init__()
        self.model_head = model_head
        self.predictor = prediction_head
        self.freeze_head = freeze_head

    def forward(self, data):
        node_emb = self.model_head(data.x, data.edge_index)
        return self.predictor(node_emb, data.batch, data.desc)

    def save(self, experiment_dir: str):
        os.makedirs(experiment_dir, exist_ok=True)
        torch.save(self.model_head.state_dict(), os.path.join(experiment_dir, 'model_head.pth'))
        torch.save(self.predictor.state_dict(), os.path.join(experiment_dir, 'predictor.pth'))

    @classmethod
    def load(cls, config: ModelConfig):
        model_head = GATStack(
            in_dim=config.model_head.in_dim,
            hidden_dim=config.model_head.hidden_dim,
            heads=config.model_head.heads,
            dropout=config.model_head.dropout,
            attn_dropout=config.model_head.attn_dropout,
        )
        if config.model_head_path is not None and os.path.exists(config.model_head_path):
            model_head.load_state_dict(torch.load(config.model_head_path, map_location='cpu'))
        if config.freeze_head:
            for p in model_head.parameters():
                p.requires_grad = False

        pred_cfg = config.predictor
        pred_cfg.pred_input_channels = model_head.graph_output_channels
        predictor = Predictor.from_config(pred_cfg)
        if config.predictor_path is not None and os.path.exists(config.predictor_path):
            predictor.load_state_dict(torch.load(config.predictor_path, map_location='cpu'))

        return cls(model_head=model_head, prediction_head=predictor, freeze_head=config.freeze_head)

    @classmethod
    def by_name(cls, model_dir: str):
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            raw = json.load(f)
        config = ModelConfig(**raw)
        config.model_head_path = os.path.join(model_dir, 'model_head.pth')
        config.predictor_path = os.path.join(model_dir, 'predictor.pth')
        return cls.load(config)
