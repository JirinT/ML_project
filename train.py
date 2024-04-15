import os
import json
import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from cnn import CNN
from tqdm import tqdm
from test import test_model
from torchviz import make_dot
from datetime import datetime
from torchvision import models
from torch.utils.data import DataLoader

from MultiHeadNetwork import MultiHeadNetwork
from torchvision.transforms import transforms
from datasets.custom_dataset import CustomDataset
from preprocessing.simple_preprocessor import SimplePreprocessor
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
import seaborn as sns


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

def show_histogram(loader, config):
    if config["cnn"]["model"]["type"]["multihead"]:
        # now we need to count the amount of samples for each class for each head
        label_counts = [{0: 0, 1: 0, 2: 0} for _ in range(config["cnn"]["model"]["num_heads"])]
        for _, labels in loader:
            for i in range(len(label_counts)):
                for label in labels:
                    label_counts[i][label.argmax().item()] += 1
        
        labels = list(label_counts[0].keys())
        counts = list(label_counts[0].values())
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(np.array(labels) - width, counts, width, label='Head 1')
        counts = list(label_counts[1].values())
        ax.bar(np.array(labels), counts, width, label='Head 2')
        counts = list(label_counts[2].values())
        ax.bar(np.array(labels) + width, counts, width, label='Head 3')
        counts = list(label_counts[3].values())
        ax.bar(np.array(labels) + 2*width, counts, width, label='Head 4')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Low', 'Good', 'High'])
        ax.legend()
        plt.grid(True)
        plt.title("hovnous")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.join(config["general"]["histogram_path"], f"histogram_multihead_{timestamp}.png"))

    else:
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

def create_folders_for_logging(config):
    now = datetime.now()
    now_formated = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_folder_training = os.path.join(config["general"]["log_path"], now_formated)
    os.makedirs(log_folder_training, exist_ok=True)
    plot_folder_training = os.path.join(config["general"]["plot_path"], now_formated)
    os.makedirs(plot_folder_training, exist_ok=True)
    model_folder_training = os.path.join(config["general"]["model_path"], now_formated)
    os.makedirs(model_folder_training, exist_ok=True)
    return log_folder_training, plot_folder_training, model_folder_training

def normalize_data(train_subset, batch_size, shuffle, dataset, num_workers, config):

    def get_mean_std(loader):
        mean = 0
        std = 0
        num_pixels = 0
        for images, _ in loader:
            mean += torch.mean(images)
            std += torch.std(images)
            num_pixels += images.numel()
        mean /= num_pixels
        std /= num_pixels
        return mean, std

    log_dir = config["cnn"]["training"]["normalization"]["log_path"]
    file_name = "mean_std.json"
    file_path = os.path.join(log_dir, file_name)

    train_loader_for_normalization = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.custom_collate_fn, num_workers=num_workers)
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
    test_subset.dataset.transform = transform

    return train_subset, val_subset, test_subset

def create_weighted_sampler(train_subset, num_classes):
    # Calculate the class frequencies
    label_counts = [0] * num_classes # [0, 0, 0] - low, good, high
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
    return sampler, shuffle

def apply_regularization(model, loss, lambda_regularization, config):
    if config["cnn"]["model"]["regularization"]["lasso"]:
        p = 1
    elif config["cnn"]["model"]["regularization"]["ridge"]:
        p = 2
    # L1 regularization term - LASSO
    l1 = torch.tensor(0.)
    for param in model.parameters():
        l1 += torch.norm(param, p=1)

    loss += lambda_regularization * l1
    return loss

def one_hot_encode(tensor):
    # Find the index of the maximum value
    max_index = tensor.argmax()
    # Create a new tensor with zeros and a 1 at the max_index
    one_hot = torch.zeros_like(tensor)
    one_hot[max_index] = 1

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
        
def compute_losses(model, images, labels, config, criterion, device):
    x1, x2, x3, x4 = model(images) # x1, x2, x3, x4 are outputs of last linear layer - raw data
    raw_predictions = [x1, x2, x3, x4] # raw_predictions are outputs of last linear layer, before LogSoftMax

    pred_heads_log_prob = [nn.LogSoftmax(dim=1)(x) for x in raw_predictions] # pred_heads_log_prob are the log probabilities
    pred_heads = decode_predictions(pred_heads_log_prob) # pred_heads is now a list of 4 tensors of shape (batch_size x 3) and contains [0,1,0] for example
    pred_heads = [x.to(device) for x in pred_heads]

    # Compute the loss for each head and update the head parameters:
    if config["cnn"]["training"]["loss_function"] != 1:
        losses = [criterion(x, labels[:,i*3:(i+1)*3]) for i, x in enumerate([x1, x2, x3, x4])]
    else:
        losses = [criterion(x, labels[:,i*3:(i+1)*3].argmax(1)) for i, x in enumerate([x1, x2, x3, x4])]
    total_loss = sum(losses)

    return total_loss, losses, pred_heads

def train():
    # Train the model
    total_step_train = len(train_loader.dataset)
    total_step_val = len(val_loader.dataset)
    best_acc = -1 # yea minus one
    start_time = time.time()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        model.train() # set model to train mode

        totalTrainLoss = 0
        totalValLoss = 0
        trainCorrect = 0 # trainCorrect is the number of correctly predicted samples in total
        valCorrect = 0 # valCorrect is the number of correctly predicted samples in total
        trainCorrect_total = 0 # trainCorrect_total is the number of correctly predicted samples for all 4 heads
        valCorrect_total = 0  # valCorrect_total is the number of correctly predicted samples for all 4 heads

        trainCorrect_list = [0, 0, 0, 0] # trainCorrect_list is the number of correctly predicted samples for each head
        valCorrect_list = [0, 0, 0, 0] # valCorrect_list is the number of correctly predicted samples for each head
        heads_train_acc = [0, 0, 0, 0] # heads_train_acc is the accuracy for each head
        heads_val_acc = [0, 0, 0, 0] # heads_val_acc is the accuracy for each head
        totalValLoss_heads = [0, 0, 0, 0] # totalValLoss_heads is the total loss for each head
        totalTrainLoss_heads = [0, 0, 0, 0] # totalTrainLoss_heads is the total loss for each head

        for train_idx, (images, labels) in tqdm(enumerate(train_loader), desc="Processing Samples", total=len(train_loader)):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            if config["cnn"]["model"]["type"]["multihead"]:
                # Compute the sum of losses in heads and update the backbone:
                total_loss, losses, pred_heads = compute_losses(model, images, labels, config, criterion, device)
                # Apply regularization
                if config["cnn"]["model"]["regularization"]["use"]:
                    loss = apply_regularization(model, loss, lambda_regularization, config)
                
                optimizer.zero_grad() # zero the gradients
                total_loss.backward() # backpropagation through the whole network
                optimizer.step() # update the weights

                totalTrainLoss += total_loss.item()
                for i in range(len(trainCorrect_list)):
                    trainCorrect_list[i] += (pred_heads[i].argmax(1) == labels[:,3*i:3*(i+1)].argmax(1)).type(torch.float).sum().item()
                    totalTrainLoss_heads[i] += losses[i].item()
                comparison = torch.all(torch.cat(pred_heads, dim=1) == labels, dim=1)
                trainCorrect_total += torch.sum(comparison).item()

            else:
                pred = model(images)
                loss = criterion(pred, labels) # calculate the loss for the current batch
                if config["cnn"]["model"]["regularization"]["use"]: # Regularization
                    loss = apply_regularization(model, loss, lambda_regularization, config)

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

        # Validate the model
        model.eval() 
        with torch.no_grad():
            for (images, labels) in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                if config["cnn"]["model"]["type"]["multihead"]:
                    total_loss, losses, pred_heads = compute_losses(model, images, labels, config, criterion, device)

                    totalValLoss += total_loss.item()
                    for i in range(len(valCorrect_list)):
                        valCorrect_list[i] += (pred_heads[i].argmax(1) == labels[:,3*i:3*(i+1)].argmax(1)).type(torch.float).sum().item()
                        totalValLoss_heads[i] += losses[i].item()

                    comparison = torch.all(torch.cat(pred_heads, dim=1) == labels, dim=1)
                    valCorrect_total += torch.sum(comparison).item()

                else:
                    pred = model(images)
                    loss = criterion(pred, labels) # calculate the loss for the current batch
                    totalValLoss += loss.item()
                    valCorrect += (pred.argmax(1) == labels.argmax(1)).type(
                    torch.float).sum().item()

        avgTrainLoss = totalTrainLoss / total_step_train
        avgValLoss = totalValLoss / total_step_val
        avgValLoss_heads = [total / total_step_val for total in totalValLoss_heads]
        avgTrainLoss_heads = [total / total_step_train for total in totalTrainLoss_heads]

        if config["cnn"]["model"]["type"]["multihead"]:
            # train accuracies:
            train_acc = trainCorrect_total / total_step_train # train_acc is the overall accuracy predicting all 4 heads
            # train accuracies for each head:
            for i in range(len(trainCorrect_list)):
                heads_train_acc[i] = trainCorrect_list[i] / total_step_train

            # val accuracies:
            val_acc = valCorrect_total / total_step_val # val_acc is the overall accuracy predicting all 4 heads
            # val accuracies for each head:
            for i in range(len(valCorrect_list)):
                heads_val_acc[i] = valCorrect_list[i] / total_step_val

        else:
            train_acc = trainCorrect / total_step_train
            val_acc = valCorrect / total_step_val

        if val_acc > best_acc:
            best_acc = valCorrect
            best_model = copy.deepcopy(model)

        # Logs
        loss_dict["train_loss"].append(avgTrainLoss)
        loss_dict["train_acc"].append(train_acc)
        loss_dict["val_loss"].append(avgValLoss)
        loss_dict["val_acc"].append(val_acc)
        with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
            file.write(f"Epoch: {epoch+1}/{num_epochs}\n")
            file.write(f"\tTrain loss: {avgTrainLoss:.4f}, Val loss: {avgValLoss:.4f}\n")
            file.write(f"\tTrain accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}\n")
            if config["cnn"]["model"]["type"]["multihead"]:
                for i in range(len(heads_train_acc)):
                    file.write(f"\t\tHead {i+1}:\n")
                    file.write(f"\t\t\tTrain loss: {avgTrainLoss_heads[i]:.4f}, Val loss: {avgValLoss_heads[i]:.4f}\n")
                    file.write(f"\t\t\tTrain accuracy: {heads_train_acc[i]:.4f}, Val accuracy: {heads_val_acc[i]:.4f}\n")
            file.write(f"\tTime elapsed: {time.time() - start_time:.2f} seconds\n")

    return best_model

# ------------------------- MAIN -------------------------------
if __name__ == "__main__":
    config = json.load(open("config.json"))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If using a GPU, clear the cache:
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()

    print("Used device is: ", device)

    # define the active user and his data path
    user = config["active_user"]
    data_path = config["general"]["data_paths"][user]

    # Create folders for logging
    log_folder_training, plot_folder_training, model_folder_training = create_folders_for_logging(config)

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
    lambda_regularization = config["cnn"]["model"]["regularization"]["lambda"]
    num_workers = config["cnn"]["training"]["num_workers"]
    num_classes = config["cnn"]["model"]["num_classes"]

    # Initialize dataset and data loader
    transform = transforms.Compose([
        SimplePreprocessor(
        width=config["preprocessor"]["resize"]["width"], 
        height=config["preprocessor"]["resize"]["height"]
        ),
        transforms.ToTensor()
    ])
    print("Loading dataset...")
    dataset = CustomDataset(data_path, transform=transform)

    num_samples_train = int(train_split * len(dataset))
    num_samples_val = int(val_split * len(dataset))
    num_samples_test = int(test_split * len(dataset))

    (train_set, val_set, test_set) = random_split(dataset, [num_samples_train, num_samples_val, num_samples_test],
                                       generator=torch.Generator().manual_seed(config["cnn"]["training"]["seed"]))

    num_samples_train_subset = int(config["cnn"]["training"]["num_samples_subset"] * train_split)
    num_samples_val_subset = int(config["cnn"]["training"]["num_samples_subset"] * val_split)
    num_samples_test_subset = int(config["cnn"]["training"]["num_samples_subset"] * test_split)

    train_subset = Subset(train_set, range(num_samples_train_subset))
    val_subset = Subset(val_set, range(num_samples_val_subset))
    test_subset = Subset(test_set, range(num_samples_test_subset))

    # Normalize the data
    if config["cnn"]["training"]["normalization"]["use"]:
        print("Normalization started...")
        train_subset, val_subset, test_subset = normalize_data(train_subset, batch_size, shuffle, dataset, num_workers, config)

    # Create weighted random sampler
    if config["cnn"]["training"]["use_weighted_rnd_sampler"]:
        print("Creating weighted random sampler...")
        sampler, shuffle = create_weighted_sampler(train_subset, num_classes)
    else:
        sampler = None
        shuffle = config["cnn"]["training"]["shuffle"]

    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.custom_collate_fn, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.custom_collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.custom_collate_fn, num_workers=num_workers)
    
    # Show histogram of the labels:
    if config["general"]["log_histograms"]:
        print("Creating histograms...")
        show_histogram(train_loader, config)

    # Choose and Initialize a model
    print("Initializing model...")
    if config["cnn"]["model"]["type"]["simple_cnn"]:
        model = CNN(config=config).to(device)

    elif config["cnn"]["model"]["type"]["resnet18"]:
        model = models.resnet18(weights=None, num_classes=num_classes).to(device)

    elif config["cnn"]["model"]["type"]["resnet34"]:
        model = models.resnet34(weights=None, num_classes=num_classes).to(device) 

    elif config["cnn"]["model"]["type"]["multihead"]:
        shared_backbone = models.resnet18(weights=None, num_classes=num_classes).to(device)
        # feature extraction - getting the output layer after convolution layers:
        return_nodes = {
        "avgpool": "AdaptiveAvgPool2d(output_size=(1, 1))"
        }
        backbone_last_layer = create_feature_extractor(shared_backbone, return_nodes=return_nodes) # the backbone_last_layer is the output of the last convolutional layer which we feed into each head
        model = MultiHeadNetwork(config, backbone_last_layer).to(device)

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

    criterion = loss_functions[config["cnn"]["training"]["loss_function"]].to(device)
    optimizer = optimizer[config["cnn"]["training"]["optimizer"]]

    loss_dict = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Train the model
    print("Training started!")
    best_model = train()
    # save the model
    if config["general"]["save_model"]:
        ("saving the model...")
        torch.save(best_model, os.path.join(model_folder_training, "model.pth"))

    # Test the model
    print("Testing started...")
    if config["cnn"]["model"]["type"]["multihead"]:
        test_accuracy, heads_test_acc = test_model(best_model, test_loader, device)
        with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
            file.write(f"\tTest accuracy: {test_accuracy * 100:.2f}%\n")
            for i in range(len(heads_test_acc)):
                file.write(f"\t\tTest accuracy head {i+1}: {heads_test_acc[i] * 100:.2f}%\n")
    else:
        test_accuracy = test_model(best_model, test_loader, device)
        with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
            file.write(f"\tTest accuracy: {test_accuracy * 100:.2f}%")

    # Visualize the network
    if config["general"]["show_net_structure"]:    
        print("Visualizing the network...")
        images, _ = next(iter(val_loader))
        os.environ["PATH"] += os.pathsep + config["cnn"]["visualization"]["graphviz_path"]
        images = images.to(device)
        dot = make_dot(model(images), params=dict(best_model.named_parameters()))
        dot.render(os.path.join(plot_folder_training, "network_graph"), format="png")

    # plot the training loss and accuracy
    print("Plotting learning curve")
    plot_learning_curve(loss_dict, plot_folder_training)

    # Create the confusion matrix
    if config["general"]["log_confusion_matrix"]:
        "Creating the confusion matrix..."
        y_true = [[] for _ in range(config["cnn"]["model"]["num_heads"])]
        y_pred = [[] for _ in range(config["cnn"]["model"]["num_heads"])]

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = best_model(images)
            
            for i, output in enumerate(outputs):
                _, predicted = torch.max(output, 1)
                y_true[i].extend(labels[:, i].tolist())
                y_pred[i].extend(predicted.tolist())

        # Create a confusion matrix for each output head
        for i in range(4):
            cm = confusion_matrix(y_true[i], y_pred[i], labels=[0, 1, 2])

            # Plot the confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix for output head {i}")
            plt.savefig(os.path.join(plot_folder_training, f"confusion_matrix_{i}.png"))
            plt.close()

    print("All done!")