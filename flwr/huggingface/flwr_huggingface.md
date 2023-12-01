## Usage 

### `server.py` 

```
--checkpoint CHECKPOINT
					Checkpoint to use. Default: distilbert-base-    
					uncased
--num_rounds NUM_ROUNDS
					Number of rounds. Default: 4
--server_address SERVER_ADDRESS
					Server address. Default:
--fraction_fit FRACTION_FIT
					Fraction of available clients used during fit.  
					Default: 1.0
--fraction_evaluate FRACTION_EVALUATE
					Fraction of available clients used during       
					evaluation. Default: 0.2
--min_fit_clients MIN_FIT_CLIENTS
					Minimum number of clients used during fit.      
					Default: 1
--min_evaluate_clients MIN_EVALUATE_CLIENTS
					Minimum number of clients used during
					evaluation. Default: 1
--min_available_clients MIN_AVAILABLE_CLIENTS
					Minimum number of available clients for each    
					round. Default: 1
--save_path SAVE_PATH
					Path to save the model. Default: model
--batch_size BATCH_SIZE
					Batch size for training. Default: 16
--local_epochs LOCAL_EPOCHS
					Number of local epochs for training. Default:   
					1
--room_id ROOM_ID     Room ID for logging. Default: VVVVV
--log_server LOG_SERVER
					Address of the log server. Default: None  
```

### `client.py` 

```
--checkpoint CHECKPOINT
					Checkpoint to use. Default: distilbert-base-uncased
--server_address SERVER_ADDRESS
					Server address. Default: localhost:8080
--device DEVICE     
					Device to use for training. Default: cuda:0
--dataset DATASET
					Choose train dataset: imdb; rotten_tomatoes Default: imdb
```

## Logger Format

`server.py` 실행 시 room_id와 log_server specify 하면 아래 형식의 로그 remote server로 보냄 

```json
{
    "identifier": "room_id",
    "levelname": "INFO",
    "name": "flwr",
    "asctime": "2022-03-01 12:00:00,000",
    "filename": "server.py",
    "lineno": 123,
    "message": "Server started"
}
```
