import json
import os
import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import transforms, datasets, models
from datasets.custom_dataset import CustomDataset
from preprocessing.simple_preprocessor import SimplePreprocessor
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.autograd import Variable
from datetime import datetime
from tqdm import tqdm

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

def get_mean_std(loader):
    mean = 0
    std = 0
    num_pixels = 0
    for images, _ in loader:
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()
        num_pixels += images.size(0) * images.size(2) * images.size(3)
    mean /= num_pixels
    std /= num_pixels
    return mean, std

def create_folders_for_logging(config):
    now = datetime.now()
    now_formated = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_folder_training = os.path.join(config["general"]["log_path"], now_formated)
    os.makedirs(log_folder_training, exist_ok=True)
    plot_folder_training = os.path.join(config["general"]["plot_path"], now_formated)
    os.makedirs(plot_folder_training, exist_ok=True)
    model_folder_training = os.path.join(config["general"]["model_path"], now_formated)
    os.makedirs(plot_folder_training, exist_ok=True)
    return log_folder_training, plot_folder_training, model_folder_training

def show_histogram(loader, config):
    label_counts = {0: 0, 1: 0, 2: 0}
    for _, labels in loader:
        for label in labels:
            label_counts[label.argmax().item()] += 1

    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.bar(labels, counts, color='skyblue')
    plt.grid(True)
    plt.xticks([0, 1, 2], ['Low', 'Good', 'High'])
    plt.title("Z-offset")
    plt.ylabel("Amount of samples")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.join(config["general"]["histogram_path"], f"histogram_separate_{timestamp}.png"))

config = json.load(open("config.json")) # load the configuration file

user = config["active_user"] # define the active user
data_path = config["general"]["data_paths"][user] # dataset path

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Used device is: ", device)

shuffle = config["cnn"]["training"]["shuffle"] 
BATCH_SIZE = config["cnn"]["training"]["batch_size"]
NUM_WORKERS = config["cnn"]["training"]["num_workers"]
NUM_SUBSET = config["cnn"]["training"]["num_samples_subset"]
NUM_EPOCHS = config["cnn"]["training"]["num_epochs"]
NUM_CLASSES = config["cnn"]["model"]["num_classes"]
LEARNING_RATE = config["cnn"]["training"]["learning_rate"]

# Create folders for logging
log_folder_training, plot_folder_training, model_folder_training = create_folders_for_logging(config)

# Initialize dataset and data loader
transform = transforms.Compose([
    SimplePreprocessor(
    width=config["preprocessor"]["resize"]["width"], 
    height=config["preprocessor"]["resize"]["height"]
    ),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
val_set = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)

num_samples_train_subset = int(NUM_SUBSET * config["cnn"]["training"]["train_split"])
num_samples_val_subset = int(NUM_SUBSET * config["cnn"]["training"]["val_split"])

train_subset = Subset(train_set, range(num_samples_train_subset))
val_subset = Subset(val_set, range(num_samples_val_subset))

# Normalization of the images
if config["cnn"]["training"]["normalization"]["use"]:
    print("Normalization started!")
    log_dir = config["cnn"]["training"]["normalization"]["log_path"]
    file_name = "mean_std.json"
    file_path = os.path.join(log_dir, file_name)

    train_loader_for_normalization = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=train_set.custom_collate_fn, num_workers=NUM_WORKERS)
    if config["cnn"]["training"]["normalization"]["compute_new_mean_std"] or not os.path.exists(file_path):
        mean, std = get_mean_std(train_loader_for_normalization)

        mean_std_dict = {
            "mean": mean.tolist(),
            "std": std.tolist()
        }

        os.makedirs(log_dir, exist_ok=True) # Check if the directory exists and create it if it doesn't
        with open (file_path, "w+") as file: # Save the mean and std to a file
            json.dump(mean_std_dict, file)

    else:
        with open(file_path, "r") as file:
            mean_std_dict = json.load(file)
            mean = mean_std_dict["mean"]
            std = mean_std_dict["std"]

    # include the normalization transform in the transform pipeline
    transform = transforms.Compose([
        SimplePreprocessor(
        width=config["preprocessor"]["resize"]["width"], 
        height=config["preprocessor"]["resize"]["height"]
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) # normalize the images
    ])
    # Apply the transform to the datasets
    train_subset.dataset.transform = transform
    val_subset.dataset.transform = transform
    print("Normalization finished!")

# WeightedRandomSampler for balancing dataset
if config["cnn"]["training"]["use_weighted_rnd_sampler"]:
    print("WeightedRandomSampler started!")
    label_counts = [0] * config["cnn"]["model"]["num_classes"] # [0, 0, 0] - low, good, high
    for _, label in train_subset:
        label_counts[label.argmax().item()] += 1 

    # Calculate the weight for each sample based on its class
    weights = []
    for _, label in train_subset:
        label = label.argmax().item()
        weights.append(1.0 / label_counts[label])

    # Create a WeightedRandomSampler with the calculated weights
    sampler = WeightedRandomSampler(weights, len(weights), replacement=False)
    shuffle = False
    print("WeightedRandomSampler finished!")
else:
    sampler = None
    shuffle = config["cnn"]["training"]["shuffle"]

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS, sampler=sampler, collate_fn=train_set.custom_collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS, collate_fn=val_set.custom_collate_fn)

# Show histogram of the labels:
if config["general"]["log_histograms"]:
    show_histogram(train_loader, config)

dataset_loaders = {
    'train': train_loader,
    'val': val_loader
    }

loss_dict = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

classes = train_subset.classes # get the class names

def train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS):
    print("Training started!")
    since = time.time()

    best_model = model
    best_acc = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        totalLoss = {
            "train": 0,
            "val": 0,
            }
        
        totalCorrect = {
            "train": 0,
            "val": 0,
            }
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval() # Set model to val mode

            # Iterate over data.
            for (images, labels) in tqdm(dataset_loaders[phase], desc="Processing Samples", total=len(dataset_loaders[phase])):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                try:
                    totalLoss += loss.item()
                    totalCorrect += torch.sum(preds == labels.data)
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

            totalLoss[phase] = totalLoss
            totalCorrect[phase] = totalCorrect
        
        avgTrainLoss = totalLoss["train"] / len(train_loader)
        avgValLoss = totalLoss["val"] / len(val_loader)
        trainCorrect = totalCorrect["train"] / len(train_loader) # trainCorrect is train accuracy
        valCorrect = totalCorrect["val"] / len(val_loader) # valCorrect is val accuracy

        if valCorrect > best_acc:
            best_acc = valCorrect
            best_model = copy.deepcopy(model)
            print('new best accuracy = ', best_acc)

        loss_dict["train_loss"].append(avgTrainLoss)
        loss_dict["train_acc"].append(trainCorrect)
        loss_dict["val_loss"].append(avgValLoss)
        loss_dict["val_acc"].append(valCorrect)
        with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
            file.write(f"Epoch: {epoch+1}/{num_epochs}\n")
            file.write(f"Train loss: {avgTrainLoss:.4f}, Val loss: {avgValLoss:.4f}\n")
            file.write(f"Train accuracy: {trainCorrect:.4f}, Val accuracy: {valCorrect:.4f}\n")
            file.write(f"Time elapsed: {time.time() - since:.2f} seconds\n")
    
    print("Training finished!")
    return best_model


# DEFINE THE MODEL:
model = models.resnet50(pretrained=True) # change to False if we want to train weight and biases from scratch

# Define loss function and optimizer
loss_functions = {
    "1": nn.CrossEntropyLoss(),
    "2": nn.MSELoss(),
    "3": nn.L1Loss(),
    "4": nn.NLLLoss()
}
optimizer = {
    "1": optim.Adam(model.parameters(), lr=LEARNING_RATE),
    "2": optim.SGD(model.parameters(), lr=LEARNING_RATE)
}

criterion = loss_functions[config["cnn"]["training"]["loss_function"]]
optimizer = optimizer[config["cnn"]["training"]["optimizer"]]

num_features = model.fc.in_features # get the number of input features for the last layer
model.fc = nn.Linear(num_features, NUM_CLASSES) # change the last layer to have the same number of classes as the dataset

if device.type == 'cuda':
    criterion.cuda()
    model.cuda()

# Train the model
model = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)

# plot the training loss and accuracy
plot_learning_curve(loss_dict, plot_folder_training)

# save the model
if config["general"]["save_model"]:
    torch.save(model, os.path.join(model_folder_training, "model.pth"))