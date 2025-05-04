# PyTorchDBCheckpoint
Checkpoint PyTorch training runs into your database.

Save model weights, optimizer state and metrics at any time. This library allows you to save your model and not worry about .pt files scattered across your file system.

## Quickstart
* Set up database schema (DDL available in ```src/ddl``` folder)
* Import ```DefaultCheckpointer``` from the package
* Configure ```DefaultCheckpointer``` class by specifying a handler and path to a config file
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
* Better error handling
* Toggle for SSL mode