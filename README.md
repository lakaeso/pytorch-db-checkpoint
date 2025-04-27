# pytorch_db_checkpoint
Checkpoint PyTorch training runs into PostgreSQL database.

Save model weights, optimizer state and metrics at any time. This library allows you to save your model and not worry about .pt files scattered across your file system.

## Quickstart
* Set up database schema (DDL available in ```src/ddl``` folder)
* Import DBHandler from the package
* Instantiate DBHandler class with a path to config file
* Use available methods to save or load model and optim state

## Config file example - database.ini

```
[postgresql]
host=your_host
database=your_database
user=your_user
password=your_password
port=your_port
```

#### Roadmap
* Support for MongoDB
* Add option for users to write own classes for data manipulation
* Add decorators to abstract use of connections and cursors