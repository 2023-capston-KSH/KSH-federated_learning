import flwr as fl
from transformers import AutoModelForSequenceClassification
import argparse
import os
import numpy as np
import os
import torch
import os
from typing import Dict, Optional, OrderedDict, Tuple, List, Union
from sklearn.metrics import precision_recall_fscore_support
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, Scalar


def fit_config(server_round: int, batch_size: int, local_epochs: int):
    """Return training configuration dict for each round."""

    config = {
        "batch_size": batch_size,
        "local_epochs": local_epochs,
    }
    return config


class FedAvgWithModel(fl.server.strategy.FedAvg):
    def __init__(self, model, save_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.save_path = save_path

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

            # Save the model in SafeTensors format
            self.model.save_pretrained(f"{self.save_path}/round_{server_round}", safe=True)
        return aggregated_parameters, aggregated_metrics


def main(args):
    # Define model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

    # Define strategy
    strategy = FedAvgWithModel(
        model,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        on_fit_config_fn=lambda r: fit_config(r, args.batch_size, args.local_epochs, args.checkpoint),
        save_path=args.save_path,
    )

    # Start server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="distilbert-base-uncased",
        required=False,
        help="Checkpoint to use for training. Default: distilbert-base-uncased",
	)
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=4,
        required=False,
        help="Number of rounds. Default: 4",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        required=False,
        help="Server address. Default:",
    )
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=1.0,
        required=False,
        help="Fraction of available clients used during fit. Default: 1.0",
    )
    parser.add_argument(
        "--fraction_evaluate",
        type=float,
        default=0.2,
        required=False,
        help="Fraction of available clients used during evaluation. Default: 0.2",
    )
    parser.add_argument(
        "--min_fit_clients",
        type=int,
        default=1,
        required=False,
        help="Minimum number of clients used during fit. Default: 1",
    )
    parser.add_argument(
        "--min_evaluate_clients",
        type=int,
        default=1,
        required=False,
        help="Minimum number of clients used during evaluation. Default: 1",
    )
    parser.add_argument(
        "--min_available_clients",
        type=int,
        default=1,
        required=False,
        help="Minimum number of available clients for each round. Default: 1",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="model",
        required=False,
        help="Path to save the model. Default: model",
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

    main(args=parser.parse_args())
