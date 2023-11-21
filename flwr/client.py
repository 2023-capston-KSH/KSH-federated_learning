import argparse
import random
import warnings
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
from datasets import load_dataset
from evaluate import load as load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

warnings.filterwarnings("ignore", category=UserWarning)


def load_data(CHECKPOINT, batch_size=32):
	"""Load IMDB data (training and eval)"""
	raw_datasets = load_dataset("imdb")
	raw_datasets = raw_datasets.shuffle(seed=42)

	tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

	def tokenize_function(examples):
		return tokenizer(examples["text"], truncation=True)

	# random 100 samples
	population = random.sample(range(len(raw_datasets["train"])), 10)
	population2 = random.sample(range(len(raw_datasets["test"])), 10)

	tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
	tokenized_datasets["train"] = tokenized_datasets["train"].select(population)
	tokenized_datasets["test"] = tokenized_datasets["test"].select(population2)

	tokenized_datasets = tokenized_datasets.remove_columns("text")
	tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	trainloader = DataLoader(
		tokenized_datasets["train"],
		shuffle=True,
		batch_size=batch_size,
		collate_fn=data_collator,
	)

	testloader = DataLoader(tokenized_datasets["test"], batch_size=32, collate_fn=data_collator)

	return trainloader, testloader


def train(net, trainloader, epochs, batch_size, DEVICE):
	optimizer = AdamW(net.parameters(), lr=5e-5)
	net.train()
	for _ in range(epochs):
		for batch in trainloader:
			batch = {k: v.to(DEVICE) for k, v in batch.items()}
			outputs = net(**batch)
			loss = outputs.loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()


def test(net, testloader, DEVICE):
	metric = load_metric("accuracy")
	loss = 0
	net.eval()
	for batch in testloader:
		batch = {k: v.to(DEVICE) for k, v in batch.items()}
		with torch.no_grad():
			outputs = net(**batch)
		logits = outputs.logits
		loss += outputs.loss.item()
		predictions = torch.argmax(logits, dim=-1)
		metric.add_batch(predictions=predictions, references=batch["labels"])
	loss /= len(testloader.dataset)
	accuracy = metric.compute()["accuracy"]
	return loss, accuracy


def main(args):
	net = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=2).to(args.device)

	trainloader, testloader = load_data(CHECKPOINT=args.checkpoint)

	# Flower client
	class IMDBClient(fl.client.NumPyClient):
		def get_parameters(self, config):
			return [val.cpu().numpy() for _, val in net.state_dict().items()]

		def set_parameters(self, parameters):
			params_dict = zip(net.state_dict().keys(), parameters)
			state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
			net.load_state_dict(state_dict, strict=True)

		def fit(self, parameters, config):
			self.set_parameters(parameters)

			epochs: int = config["local_epochs"]
			batch_size: int = config["batch_size"]

			print("Training Started...")
			train(net, trainloader, epochs=epochs, batch_size=batch_size, DEVICE=args.device)
			print("Training Finished.")
			return self.get_parameters(config={}), len(trainloader), {}

		def evaluate(self, parameters, config):
			self.set_parameters(parameters)
			loss, accuracy = test(net, testloader, DEVICE=args.device)
			return float(loss), len(testloader), {"accuracy": float(accuracy)}

	# Start client
	fl.client.start_numpy_client(server_address=args.server_address, client=IMDBClient())


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Flower")
	parser.add_argument(
		"--checkpoint",
		type=str,
		default="distilbert-base-uncased",
		required=False,
		help="Checkpoint to use. Default: distilbert-base-uncased",
	)
	parser.add_argument(
		"--server_address",
		type=str,
		default="127.0.0.1:8080",
		required=False,
		help="Server address. Default:"
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda:0" if torch.cuda.is_available() else "cpu",
		required=False,
		help="Device to use for training. Default: cuda:0",
	)

	main(args=parser.parse_args())
