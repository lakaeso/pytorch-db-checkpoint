import pymongo
from configparser import ConfigParser
from pathlib import Path
from ..utils import CheckpointData
import pickle



class MongoHandler:
    """Abstracts access to Mongo database."""

    def __init__(self, path_to_config: str | Path, section: str ='mongodb') -> None:
        """
        Inits MongoDB instance.
        
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

    def save_training_state(self, data: CheckpointData):
        """
        Saves training state into a db.

        :param CheckpointData: Data to insert.
        """
        epoch = data.epoch
        model_name = data.model_name
        model_state_dict = pickle.dumps(data.model_state_dict)
        optim_state_dict = pickle.dumps(data.optim_state_dict)
        comment = data.comment
        metrics = data.metrics

        client = pymongo.MongoClient(self._config["connectionstring"], int(self._config["port"]))
        db = client.get_database(self._config["database"])
        collection = db.get_collection(self._config["collection"])

        collection.insert_one({"epoch": epoch, "model_name": model_name, "model_state_dict": model_state_dict, "optim_state_dict": optim_state_dict, "comment": comment, "metrics": metrics})
    
    def load_training_state_last_epoch(self, model_name: str, *args, **kwargs) -> CheckpointData:
        
        client = pymongo.MongoClient(self._config["connectionstring"], int(self._config["port"]))
        db = client.get_database(self._config["database"])
        collection = db.get_collection(self._config["collection"])

        row = collection.find({"model_name":model_name}).sort({"timestamp_inserted": -1})[0]

        data = CheckpointData(
            model_name=row["model_name"], 
            epoch=row["epoch"], 
            model_state_dict=pickle.loads(row["model_state_dict"]), 
            optim_state_dict=pickle.loads(row["optim_state_dict"]),
            metrics=row["metrics"], 
            comment=row["comment"]
        )

        return data
    
    def load_training_state_last_entry(self, model_name: str, *args, **kwargs) -> CheckpointData:
        
        client = pymongo.MongoClient(self._config["connectionstring"], int(self._config["port"]))
        db = client.get_database(self._config["database"])
        collection = db.get_collection(self._config["collection"])

        row = collection.find({"model_name":model_name}).sort({"epoch": -1})[0]

        data = CheckpointData(
            model_name=row["model_name"], 
            epoch=row["epoch"], 
            model_state_dict=pickle.loads(row["model_state_dict"]), 
            optim_state_dict=pickle.loads(row["optim_state_dict"]),
            metrics=row["metrics"], 
            comment=row["comment"]
        )

        return data