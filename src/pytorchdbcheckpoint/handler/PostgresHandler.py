import psycopg2
from configparser import ConfigParser
from pathlib import Path
from .checkpoint_data import CheckpointData
import pickle
import json


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
    def save_training_state(self, data: CheckpointData , *args, **kwargs):
        """Saves ```CheckpointData``` object into a database. """
        
        # get data
        epoch = data.epoch
        model_name = data.model_name
        model_state_dict = pickle.dumps(data.model.state_dict())
        optim_state_dict = pickle.dumps(data.optim.state_dict())
        comment = data.comment
        metrics_str = json.dumps(data.metrics)

        # get cursor (db connection)
        cur = kwargs["cur"]

        # execute query
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
    def load_training_state_last_epoch(self, model_name: str, *args, **kwargs):
        
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

        return obj
    
    @_connection_decorator
    def load_training_state_last_entry(self, model_name: str, *args, **kwargs):

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

        return obj