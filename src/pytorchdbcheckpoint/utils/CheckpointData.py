from dataclasses import dataclass
import torch
from typing import Any



@dataclass
class CheckpointData:
    model_name: str
    epoch: int
    model_state_dict: dict[str, Any]
    optim_state_dict: dict[str, Any]
    metrics: dict 
    comment: str