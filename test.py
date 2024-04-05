import json
import os
import torch

from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import transforms

from datasets.custom_dataset import CustomDataset
from preprocessing.simple_preprocessor import SimplePreprocessor


def test_model(model, test_loader, device):
    """
    Test the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to test.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        device (torch.device): The device to run the model on.

    Returns:
        float: The accuracy of the model on the test dataset.
    """

    model.eval()
    with torch.no_grad():
        correct = 0
        for test_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.argmax(1)

            pred = model(images)
            correct += (pred.argmax(1) == labels).type(
            torch.float).sum().item()
        
        accuracy = correct / len(test_loader.dataset)

    return accuracy


config = json.load(open("config.json"))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the active user and his data path
user = config["active_user"]
data_path = config["general"]["data_paths"][user]

# Define hyperparameters
batch_size = config["cnn"]["training"]["batch_size"]
shuffle = config["cnn"]["training"]["shuffle"]
train_split = config["cnn"]["training"]["train_split"]
val_split = config["cnn"]["training"]["val_split"]
test_split = config["cnn"]["training"]["test_split"]

# Initialize dataset and data loader
transform = transforms.Compose([
    SimplePreprocessor(
	width=config["preprocessor"]["resize"]["width"], 
	height=config["preprocessor"]["resize"]["height"]
	),
    transforms.ToTensor()
])

dataset = CustomDataset(data_path, transform=transform)

num_samples_train = int(train_split * len(dataset))
num_samples_val = int(val_split * len(dataset))
num_samples_test = int(test_split * len(dataset))

(train_set, val_set, test_set) = random_split(dataset, [num_samples_train, num_samples_val, num_samples_test], generator=torch.Generator().manual_seed(config["cnn"]["training"]["seed"]))

num_samples_train_subset = int(config["cnn"]["training"]["num_samples_subset"] * train_split)
num_samples_val_subset = int(config["cnn"]["training"]["num_samples_subset"] * val_split)
num_samples_test_subset = int(config["cnn"]["training"]["num_samples_subset"] * test_split)

train_subset = Subset(train_set, range(num_samples_train_subset))
val_subset = Subset(val_set, range(num_samples_val_subset))
test_subset = Subset(test_set, range(num_samples_test_subset))

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=shuffle)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=shuffle)

"""
model = torch.load(os.path.join(config["general"]["model_path"], "model.pth")).to(device)

accuracy = test_model(model, test_loader, device)
print(f'Accuracy: {accuracy * 100:.2f}%')
"""