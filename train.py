import json
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import transforms
from tqdm import tqdm

from datasets.custom_dataset import CustomDataset
from preprocessing.simple_preprocessor import SimplePreprocessor
from cnn import CNN
from test import test_model


def plot_learning_curve(loss_dict, plot_folder_training):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(loss_dict["train_loss"], label="train_loss")
    plt.plot(loss_dict["val_loss"], label="val_loss")
    plt.plot(loss_dict["train_acc"], label="train_acc")
    plt.plot(loss_dict["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(plot_folder_training, "loss_plot.png"))

config = json.load(open("config.json"))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the active user and his data path
user = config["active_user"]
data_path = config["general"]["data_paths"][user]

# create folders for logging
now = datetime.now()
now_formated = now.strftime("%Y-%m-%d_%H-%M-%S")
log_folder_training = os.path.join(config["general"]["log_path"], now_formated)
os.makedirs(log_folder_training, exist_ok=True)
plot_folder_training = os.path.join(config["general"]["plot_path"], now_formated)
os.makedirs(plot_folder_training, exist_ok=True)
model_folder_training = os.path.join(config["general"]["model_path"], now_formated)
os.makedirs(plot_folder_training, exist_ok=True)

# Save the config to a text file
filename_config = os.path.join(log_folder_training, "config.txt")
with open(filename_config, 'w') as f:
    json.dump(config, f)

# Define hyperparameters
num_epochs = config["cnn"]["training"]["num_epochs"]
batch_size = config["cnn"]["training"]["batch_size"]
learning_rate = config["cnn"]["training"]["learning_rate"]
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

# Initialize model
model = CNN(config=config).to(device)

# Define loss function and optimizer
loss_functions = {
    "1": nn.CrossEntropyLoss(),
    "2": nn.MSELoss(),
    "3": nn.L1Loss(),
    "4": nn.NLLLoss()
}
optimizer = {
    "1": optim.Adam(model.parameters(), lr=learning_rate),
    "2": optim.SGD(model.parameters(), lr=learning_rate)
}

criterion = loss_functions[config["cnn"]["training"]["loss_function"]]
optimizer = optimizer[config["cnn"]["training"]["optimizer"]]


loss_dict = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

# Train the model
total_step_train = len(train_loader.dataset)
total_step_val = len(val_loader.dataset)
print("Training started!")
start_time = time.time()
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    # set model to train mode
    model.train()

    totalTrainLoss = 0
    totalValLoss = 0
    trainCorrect = 0
    valCorrect = 0
    for train_idx, (images, labels) in tqdm(enumerate(train_loader), desc="Processing Samples", total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        pred = model(images)
        loss = criterion(pred, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalTrainLoss += loss.item()
        trainCorrect += (pred.argmax(1) == labels.argmax(1)).type(
            torch.float).sum().item()

        if (train_idx+1) % config["cnn"]["training"]["print_step"] == 0:
            with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
                file.write(f"Epoch [{epoch+1}/{num_epochs}], Step [{train_idx+1}/{total_step_train}], Loss: {loss.item():.4f}\n")

    model.eval()
    with torch.no_grad():
        for val_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            loss = criterion(pred, labels)

            totalValLoss += loss.item()
            valCorrect += (pred.argmax(1) == labels.argmax(1)).type(
                torch.float).sum().item()

    avgTrainLoss = totalTrainLoss / total_step_train
    avgValLoss = totalValLoss / total_step_val
    trainCorrect = trainCorrect / total_step_train
    valCorrect = valCorrect / total_step_val
    loss_dict["train_loss"].append(avgTrainLoss)
    loss_dict["train_acc"].append(trainCorrect)
    loss_dict["val_loss"].append(avgValLoss)
    loss_dict["val_acc"].append(valCorrect)
    with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
        file.write(f"Epoch: {epoch+1}/{num_epochs}\n")
        file.write(f"Train loss: {avgTrainLoss:.4f}, Val loss: {avgValLoss:.4f}\n")
        file.write(f"Train accuracy: {trainCorrect:.4f}, Val accuracy: {valCorrect:.4f}\n")
        file.write(f"Time elapsed: {time.time() - start_time:.2f} seconds\n")

print("Training finished!")

# Test the model
test_accuracy = test_model(model, test_loader, device)
with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
    file.write(f"Test accuracy: {test_accuracy * 100:.2f}%")

# plot the training loss and accuracy
plot_learning_curve(loss_dict, plot_folder_training)

# save the model
if config["general"]["save_model"]:
    torch.save(model, os.path.join(model_folder_training, "model.pth"))