from .gat import Model as GATModel
from .gcn import Model as GCNModel
from .gin import Model as GINModel

def model_factory(name: str):
    name = name.lower()
    if name == "gat": return GATModel
    if name == "gcn": return GCNModel
    if name == "gin": return GINModel
    raise ValueError(f"Unknown model: {name}")
