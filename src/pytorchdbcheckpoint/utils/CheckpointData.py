from dataclasses import dataclass
import torch



@dataclass
class CheckpointData:
    model_name: str
    epoch: int
    model: torch.nn.Module
    optim: torch.optim.Optimizer
    metrics: dict 
    comment: str