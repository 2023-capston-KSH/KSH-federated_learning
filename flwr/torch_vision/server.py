from typing import Dict, Optional, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader

import argparse
import flwr as fl
import torch
import utils
import warnings


warnings.filterwarnings("ignore")


def fit_config(server_round: int, batch_size: int, local_epochs: int):
    """Return training configuration dict for each round."""

    config = {
        "batch_size": batch_size,
        "local_epochs": local_epochs,
    }
    return config


def evaluate_config(server_round: int, val_steps: int):
    """Return evaluation configuration dict for each round."""

    return {"val_steps": val_steps}


import os

def get_evaluate_fn(
	model: torch.nn.Module, toy: bool, valset: torch.utils.data.Subset, save_path: str
):
	"""Return an evaluation function for server-side evaluation."""

	valLoader = DataLoader(valset, batch_size=16)

	# Create directory if it does not exist
	os.makedirs(os.path.dirname(save_path), exist_ok=True)

	# The `evaluate` function will be called after every round
	def evaluate(
		server_round: int,
		parameters: fl.common.NDArray,
		config: Dict[str, fl.common.Scalar],
	) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
		# Update model with the latest parameters
		params_dict = zip(model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		model.load_state_dict(state_dict, strict=True)

		loss, accuracy = utils.test(model, valLoader)

		# Save the model after each round
		torch.save(model.state_dict(), save_path.format(server_round))

		return loss, {"accuracy": accuracy}

	return evaluate


def load_model() -> torch.nn.Module:
    """Load the EfficientNet model."""

    return utils.load_efficientnet(classes=10)


def load_data(
    toy: bool, data: str, data_dir: str
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load the training and validation data."""

    trainset, _, _ = utils.load_data(data, data_dir)
    n_train = len(trainset)
    if toy:
        # use only 10 samples as validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
    else:
        # Use the last 5k training examples as a validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))
    return trainset, valset


def create_strategy(
    model: torch.nn.Module,
    trainset: torch.utils.data.Dataset,
    valset: torch.utils.data.Dataset,
    args: argparse.Namespace,
) -> fl.server.strategy.FedAvg:
    """Create the FedAvg strategy."""

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    save_path = args.save_path

    return fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        evaluate_fn=get_evaluate_fn(model, args.toy, valset, save_path),
        on_fit_config_fn=lambda r: fit_config(r, args.batch_size, args.local_epochs),
        on_evaluate_config_fn=lambda r: evaluate_config(r, args.val_steps),
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )


def main(args: argparse.Namespace) -> None:
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    model = load_model()
    trainset, valset = load_data(args.toy, args.dataset, args.data_dir)
    strategy = create_strategy(model, trainset, valset, args)

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use only 10 datasamples for validation. \
			Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./model/model_round_{}.pt",
        required=False,
        help="Path to save the model after each round. Default: ./model/model_round_{}.pt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        required=False,
        help="Batch size for training. Default: 16",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        required=False,
        help="Number of local epochs for training. Default: 1",
    )
    parser.add_argument(
        "--val_steps",
        type=int,
        default=5,
        required=False,
        help="Number of local evaluation steps on each client. Default: 5",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=4,
        required=False,
        help="Number of rounds for federated learning. Default: 4",
    )
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=0.2,
        required=False,
        help="Fraction of clients used for training in each round. Default: 0.2",
    )
    parser.add_argument(
        "--fraction_evaluate",
        type=float,
        default=0.2,
        required=False,
        help="Fraction of clients used for evaluation in each round. Default: 0.2",
    )
    parser.add_argument(
        "--min_fit_clients",
        type=int,
        default=2,
        required=False,
        help="Minimum number of clients used for training in each round. Default: 2",
    )
    parser.add_argument(
        "--min_evaluate_clients",
        type=int,
        default=2,
        required=False,
        help="Minimum number of clients used for evaluation in each round. Default: 2",
    )
    parser.add_argument(
        "--min_available_clients",
        type=int,
        default=10,
        required=False,
        help="Minimum number of available clients for each round. Default: 10",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        required=False,
        help="Choose any dataset from torchvision.datasets. Default: cifar10",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        required=False,
        help="Path to the directory containing the dataset. Default: ./dataset",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        required=False,
        help="Server address for the Flower server. Default: 0.0.0.0:8080",
    )
    args = parser.parse_args()
    main(args)
