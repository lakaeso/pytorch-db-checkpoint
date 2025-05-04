from pathlib import Path
from ..handler import PostgresHandler, MongoHandler



class HandlerFactory:
    """Used for fetching a handler type."""

    @staticmethod
    def get_handler(handler: str, path_to_config: str | Path, section: str):
        """Returns wanted handler based on parameter ```keyword```. Throws Exception if no matching keywork found."""
        if handler == 'postgres':
            return PostgresHandler(path_to_config, section)
        if handler == 'mongo':
            return MongoHandler(path_to_config, section)
        raise Exception(f"Error, no handler matching {handler}.")