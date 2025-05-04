import torch.nn as nn
import torch.optim as optim
import pickle
import json
from pathlib import Path
from ..handler import HandlerFactory, CheckpointData

class DefaultCheckpointer:
    """Default class used for checkpoint PyTorch training runs or experiments."""

    def __init__(self, handler: str, path_to_config: str | Path, section: str, verbose: bool = False):
        """
        Inits DefaultCheckpointer class.
        
        :param str handler: choose the available DB handler
        :param str | Path path_to_config: Path to .ini config file
        :param str section: Section in .ini config file
        """
        self.handler = HandlerFactory.get_handler(handler, path_to_config, section)
        self.verbose = verbose

        if self.verbose: 
            print(f"DefaultCheckpointer init done, {handler=}.")

    def save_training_state(self, model_name: str, epoch: int, model: nn.Module, optim: optim.Optimizer, metrics: dict = None, comment: str = None, *args, **kwargs):
        """
        Saves training state to a database.
        
        :param str model_name: Name under which model will be saved
        :param int epoch: Current epoch number
        :param nn.Module model: PyTorch model
        :param optim.Optimizer: PyTorch optimizer
        :param dict metrics: Python dictionary of training metrics (accuracy, f1 ...)
        :param str comment: Your comment, if you have any
        """
        if self.verbose: 
            print(f"DefaultCheckpointer saving traning state, {model_name=}, {epoch=}")

        model_state_dict = pickle.dumps(model.state_dict())

        optim_state_dict = pickle.dumps(optim.state_dict())

        metrics_str = json.dumps(metrics)

        data = CheckpointData(model_name=model_name, epoch=epoch, model=model, optim=optim, metrics=metrics, comment=comment)

        self.handler.save_training_state(data)
    
    def load_training_state_last_epoch(self, model_name: str, model: nn.Module, optim: optim.Optimizer | None, *args, **kwargs):
        """
        Load training state by model name and last epoch.
        
        :param str model_name: Name of the model to load
        :param nn.Module model: PyTorch model
        :param optim.Optimizer: PyTorch optimizer
        """
        if self.verbose: 
            print(f"DefaultCheckpointer loading traning state by last epoch, {model_name=}")

        obj = self.handler.load_training_state_last_epoch(model_name)

        epoch = obj[1]

        model.load_state_dict(pickle.loads(obj[3]))

        if optim is not None:
            optim.load_state_dict(pickle.loads(obj[4]))

        return epoch, model, optim
    
    def load_training_state_last_entry(self, model_name: str, model: nn.Module, optim: optim.Optimizer | None, *args, **kwargs):
        """
        Load training state by model name and last entry.
        
        :param str model_name: Name of the model to load
        :param nn.Module model: PyTorch model
        :param optim.Optimizer: PyTorch optimizer
        """
        if self.verbose: 
            print(f"DefaultCheckpointer loading traning state by last entry, {model_name=}")
        
        obj = self.handler.load_training_state_last_entry(model_name)

        epoch = obj[1]

        model.load_state_dict(pickle.loads(obj[3]))

        if optim is not None:
            optim.load_state_dict(pickle.loads(obj[4]))

        return epoch, model, optim
    