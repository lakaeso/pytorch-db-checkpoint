import psycopg2
import torch
import torch.nn as nn
import pickle
from configparser import ConfigParser


class DBHandler:

    _config = {}

    def __init__(self):
        self._config = self._load_config()

    def _load_config(self, filename='database.ini', section='postgresql'):
        parser = ConfigParser()
        parser.read(filename)
        config = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                config[param[0]] = param[1]
        else:
            raise Exception(f'Section {section} not found in the {filename} file.')
        return config
    
    def _create_connection(self):
        config = self._config
        try:
            with psycopg2.connect(**config) as conn:
                return conn
        except (psycopg2.DatabaseError, Exception) as error:
            print(error)
    
    def save_training_state(self, model: nn.Module, epoch: int, optim: torch.optim.Optimizer, loss: nn.Module):

        # TODO: check if model with the same name exists but with larger epoch number
        # ask user to overwrite? or overwrite flag??

        model_state_dict = pickle.dumps(model.state_dict())

        optim_state_dict = pickle.dumps(optim.state_dict())

        with self._create_connection() as conn:
            
            cur = conn.cursor()
            
            cur.execute("INSERT INTO training_checkpoint (epoch, model_name, model_state_dict, optim_state_dict, timestamp_inserted, comment) VALUES (%s, %s, %s, %s, current_timestamp, %s)", (epoch, "m", model_state_dict, optim_state_dict, "blah"))
            
            conn.commit()
            
            cur.close()
        
        conn.close()
    
    def load_training_state(self, model, optim):
        
        with self._create_connection() as conn:
        
            cur = conn.cursor()

            cur.execute("SELECT * FROM training_checkpoint WHERE model_name = %s ORDER BY epoch DESC", ("m"))

            obj = cur.fetchone()

            epoch = obj[1]

            model.load_state_dict(pickle.loads(obj[3]))

            optim.load_state_dict(pickle.loads(obj[4]))

            cur.close()

        conn.close()

        return epoch, model, optim
