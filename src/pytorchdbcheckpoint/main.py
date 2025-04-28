import psycopg2
import torch.nn as nn
import torch.optim as optim
import pickle
from configparser import ConfigParser
import json
from pathlib import Path


class PostgresHandler:
    """Abstracts access to PostgreSQL database."""
    _config = {}

    def __init__(self, path_to_config: str | Path, section: str ='postgresql') -> None:
        """
        Inits PostgresHandler instance.
        
        :param str | Path path_to_config: Path to config ```.ini``` file
        :param str section: Section in config file
        """
        self._config = self._load_config(path_to_config, section)

    def _load_config(self, path_to_config: str | Path, section: str) -> dict:
        """
        Loads config file from path and returns it in a form of a dictionary.
        
        :param str | Path path_to_config: Path to config ```.ini``` file
        :param str section: Section in config file
        """
        parser = ConfigParser()
        parser.read(path_to_config)
        config = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                config[param[0]] = param[1]
        else:
            raise Exception(f'Section {section} not found in the {path_to_config} file.')
        return config
    
    def _create_connection(self):
        """
        Creates ```psycopg2``` connection.
        """
        config = self._config
        try:
            with psycopg2.connect(**config) as conn:
                return conn
        except (psycopg2.DatabaseError, Exception) as error:
            print(error)
    
    def _connection_decorator(func):
        """
        Decorator for methods which access the database.

        Appends psycopg2 cursor ```cur``` object to function's kwargs.
        """
        def wrapper(self, *args, **kwargs):
            with self._create_connection() as conn:
                with conn.cursor() as cur:
                    kwargs["cur"] = cur
                    return_value = func(self, *args, **kwargs)
                    cur.close()
            return return_value
        return wrapper
    
    @_connection_decorator
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

        cur = kwargs["cur"]

        model_state_dict = pickle.dumps(model.state_dict())

        optim_state_dict = pickle.dumps(optim.state_dict())

        metrics_str = json.dumps(metrics)
            
        cur.execute(
            """
            INSERT INTO training_checkpoint 
                    (epoch, model_name, model_state_dict, optim_state_dict, timestamp_inserted, comment, metrics) 
            VALUES 
                    (%s, %s, %s, %s, current_timestamp, %s, %s)
            """, 
            (epoch, model_name, model_state_dict, optim_state_dict, comment, metrics_str)
        )
    
    @_connection_decorator
    def load_training_state_last_epoch(self, model_name: str, model: nn.Module, optim: optim.Optimizer | None, *args, **kwargs):
        """
        Load training state by model name and last epoch.
        
        :param str model_name: Name of the model to load
        :param nn.Module model: PyTorch model
        :param optim.Optimizer: PyTorch optimizer
        """

        cur = kwargs["cur"]
        
        cur.execute(
            """
            SELECT 
                * 
            FROM 
                training_checkpoint 
            WHERE 
                model_name = %s 
            ORDER BY 
                epoch DESC
            """, 
            (model_name, )
        )

        obj = cur.fetchone()

        epoch = obj[1]

        model.load_state_dict(pickle.loads(obj[3]))

        if optim is not None:
            optim.load_state_dict(pickle.loads(obj[4]))

        return epoch, model, optim
    
    @_connection_decorator
    def load_training_state_last_entry(self, model_name: str, model: nn.Module, optim: optim.Optimizer | None, *args, **kwargs):
        """
        Load training state by model name and last entry.
        
        :param str model_name: Name of the model to load
        :param nn.Module model: PyTorch model
        :param optim.Optimizer: PyTorch optimizer
        """
        
        cur = kwargs["cur"]
    
        cur.execute(
            """
            SELECT 
                * 
            FROM 
                training_checkpoint 
            WHERE 
                model_name = %s 
            ORDER BY 
                timestamp_inserted DESC
            """, 
            (model_name, )
        )

        obj = cur.fetchone()

        epoch = obj[1]

        model.load_state_dict(pickle.loads(obj[3]))

        if optim is not None:
            optim.load_state_dict(pickle.loads(obj[4]))

        return epoch, model, optim
