import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageNet, STL10, SVHN

import warnings

warnings.filterwarnings("ignore")

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(dataset_name, data_dir="./dataset"):
    """Load a dataset from the torchvision.datasets module."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )

    if dataset_name == "cifar10":
        trainset = CIFAR10(data_dir, train=True,
                           download=True, transform=transform)
        testset = CIFAR10(data_dir, train=False,
                          download=True, transform=transform)
    elif dataset_name == "cifar100":
        trainset = CIFAR100(data_dir, train=True,
                            download=True, transform=transform)
        testset = CIFAR100(data_dir, train=False,
                           download=True, transform=transform)
    elif dataset_name == "mnist":
        trainset = MNIST(data_dir, train=True,
                         download=True, transform=transform)
        testset = MNIST(data_dir, train=False,
                        download=True, transform=transform)
    elif dataset_name == "fashion_mnist":
        trainset = FashionMNIST(data_dir, train=True,
                                download=True, transform=transform)
        testset = FashionMNIST(data_dir, train=False,
                               download=True, transform=transform)
    elif dataset_name == "imagenet":
        trainset = ImageNet(data_dir, split="train",
                            download=True, transform=transform)
        testset = ImageNet(data_dir, split="val",
                           download=True, transform=transform)
    elif dataset_name == "stl10":
        trainset = STL10(data_dir, split="train",
                         download=True, transform=transform)
        testset = STL10(data_dir, split="test",
                        download=True, transform=transform)
    elif dataset_name == "svhn":
        trainset = SVHN(data_dir, split="train",
                        download=True, transform=transform)
        testset = SVHN(data_dir, split="test",
                       download=True, transform=transform)
    else:
        raise ValueError("Invalid dataset name")

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples


def load_partition(idx: int, dataset_name: str):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    trainset, testset, num_examples = load_data(dataset_name)
    n_train = int(num_examples["trainset"] / 10)
    n_test = int(num_examples["testset"] / 10)

    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition)


def train(net, trainloader, valloader, epochs, device: str = "cpu"):
    """Train the network on the training set."""
    print("Starting training...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    net.to("cpu")  # move model back to CPU

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader, steps: int = None, device: str = "cpu"):
    """Validate the network on the entire test set."""
    print("Starting evalutation...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break
    accuracy = correct / len(testloader.dataset)
    net.to("cpu")  # move model back to CPU
    return loss, accuracy


def replace_classifying_layer(efficientnet_model, num_classes: int = 10):
    """Replaces the final layer of the classifier."""
    num_features = efficientnet_model.classifier.fc.in_features
    efficientnet_model.classifier.fc = torch.nn.Linear(
        num_features, num_classes)


def load_efficientnet(entrypoint: str = "nvidia_efficientnet_b0", classes: int = None):
    """Loads pretrained efficientnet model from torch hub. Replaces final classifying
    layer if classes is specified.

    Args:
        entrypoint: EfficientNet model to download.
                    For supported entrypoints, please refer
                    https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/
        classes: Number of classes in final classifying layer. Leave as None to get the downloaded
                 model untouched.
    Returns:
        EfficientNet Model

    Note: One alternative implementation can be found at https://github.com/lukemelas/EfficientNet-PyTorch
    """
    efficientnet = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", entrypoint, pretrained=True
    )

    if classes is not None:
        replace_classifying_layer(efficientnet, classes)
    return efficientnet


def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
