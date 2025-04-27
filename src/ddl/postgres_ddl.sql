CREATE TABLE training_checkpoint (
    ID serial NOT NULL PRIMARY KEY,
    epoch INTEGER NOT NULL,
	model_name TEXT NOT NULL,
	model_state_dict bytea NOT NULL,
	optim_state_dict bytea,
	timestamp_inserted timestamp NOT NULL,
	comment TEXT,
	metrics json
);