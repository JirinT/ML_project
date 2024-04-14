import json
import os
import torch

from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import transforms
import torch.nn as nn

from datasets.custom_dataset import CustomDataset
from preprocessing.simple_preprocessor import SimplePreprocessor

def one_hot_encode(tensor):
    # Find the index of the maximum value
    max_index = tensor.argmax()
    # Create a new tensor with zeros and a 1 at the max_index
    one_hot = torch.zeros_like(tensor)
    one_hot[max_index] = 1
    # one_hot = one_hot.numpy()

    return one_hot

def decode_predictions(pred_heads_log_prob):
    prob_heads = [torch.exp(x) for x in pred_heads_log_prob]  # convert to standard probabilities
    pred_heads_list = []
    for j in range(len(prob_heads)):
        pred_head = torch.zeros((len(prob_heads[j]), 3))
        for i in range(len(prob_heads[j])):
            pred_head[i] = one_hot_encode(prob_heads[j][i])
        pred_heads_list.append(pred_head)

    return pred_heads_list

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
    correct_list = [0, 0, 0, 0]
    heads_train_acc = [0, 0, 0, 0]
    correct = 0
    testCorrect_total = 0

    model.eval()
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if config["cnn"]["model"]["type"]["multihead"]:
                x1, x2, x3, x4 = model(images) # x1, x2, x3, x4 are outputs of last linear layer - raw data
                raw_predictions = [x1, x2, x3, x4] # raw_predictions are outputs of last linear layer, before LogSoftMax

                pred_heads_log_prob = [nn.LogSoftmax(dim=1)(x) for x in raw_predictions] # pred_heads_log_prob are the log probabilities
                pred_heads = decode_predictions(pred_heads_log_prob) # pred_heads is now a list of 4 tensors of shape (batch_size x 3) and contains [0,1,0] for example
                pred_heads = [x.to(device) for x in pred_heads]

                for i in range(len(pred_heads)):
                    correct_list[i] += (pred_heads[i].argmax(1) == labels[:,i*3:(i+1)*3].argmax(1)).type(torch.float).sum().item()
                comparison = torch.all(torch.cat(pred_heads, dim=1) == labels, dim=1)
                testCorrect_total += torch.sum(comparison).item()
            else:
                pred = model(images)
                correct += (pred.argmax(1) == labels.argmax(1)).type(
                torch.float).sum().item()
        
        if config["cnn"]["model"]["type"]["multihead"]:
            for i in range(len(correct_list)):
                heads_train_acc[i] = correct_list[i] / len(test_loader.dataset)
            total_accuracy = testCorrect_total / len(test_loader.dataset)

            return total_accuracy, heads_train_acc
        else:
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