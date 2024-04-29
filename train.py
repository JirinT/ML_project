import os
import json
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from test import test_model
from torchview import draw_graph
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from datasets.custom_dataset import CustomDataset
from folder_functions import create_folders_logging
from preprocessing.simple_preprocessor import SimplePreprocessor
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import transforms
from utils import Visualization, PreprocessingUtils, ModelUtils


def train():
    # Train the model
    total_step_train = len(train_loader.dataset)
    total_step_val = len(val_loader.dataset)
    best_acc = -1 # yea minus one
    patience = 3 # early stopping - if the validation loss does not decrease for {patience} epochs, stop training
    best_val_loss = float("inf") # set the best validation loss to infinity
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
            if config["cnn"]["model"]["use_multihead"]:
                # Compute the sum of losses in heads and update the backbone:
                total_loss, losses, pred_heads = model_utils.compute_losses(model, images, labels, config, criterion)
                # Apply regularization
                if config["cnn"]["model"]["regularization"]["use"]:
                    total_loss = model_utils.apply_regularization(model, total_loss)
                
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
                    loss = model_utils.apply_regularization(model, loss)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                totalTrainLoss += loss.item()
                trainCorrect += (pred.argmax(1) == labels.argmax(1)).type(
                torch.float).sum().item()

            if (train_idx+1) % config["cnn"]["training"]["print_step"] == 0:
                with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
                    file.write(f"Epoch [{epoch+1}/{num_epochs}], Step [{train_idx+1}/{total_step_train}], Loss: {total_loss.item():.4f}\n")

        # Validate the model
        model.eval()
        with torch.no_grad():
            for (images, labels) in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                if config["cnn"]["model"]["use_multihead"]:
                    total_loss, losses, pred_heads = model_utils.compute_losses(model, images, labels, config, criterion)

                    totalValLoss += total_loss.item()
                    for i in range(len(valCorrect_list)):
                        valCorrect_list[i] += (pred_heads[i].argmax(1) == labels[:,3*i:3*(i+1)].argmax(1)).type(torch.float).sum().item()
                        totalValLoss_heads[i] += losses[i].item()

                    comparison = torch.all(torch.cat(pred_heads, dim=1) == labels, dim=1)
                    valCorrect_total += torch.sum(comparison).item()
                    if config["cnn"]["training"]["lr_scheduler"]["use"]:
                        scheduler.step()
                    

                else:
                    pred = model(images)
                    loss = criterion(pred, labels) # calculate the loss for the current batch
                    totalValLoss += loss.item()
                    valCorrect += (pred.argmax(1) == labels.argmax(1)).type(
                    torch.float).sum().item()

        avgValLoss_heads = [total / total_step_val for total in totalValLoss_heads] # average val loss for each head
        avgTrainLoss_heads = [total / total_step_train for total in totalTrainLoss_heads] # average train loss for each head
        avgHeadValLoss = sum(avgValLoss_heads) / len(avgValLoss_heads) # average val loss for all heads
        avgHeadTrainLoss = sum(avgTrainLoss_heads) / len(avgTrainLoss_heads) # average train loss for all heads

        if config["cnn"]["model"]["use_multihead"]:
            # train accuracies for each head:
            for i in range(len(trainCorrect_list)):
                heads_train_acc[i] = trainCorrect_list[i] / total_step_train
            avgHeadTrainAcc = sum(heads_train_acc) / len(heads_train_acc) # average train accuracy for all heads
            # val accuracies for each head:
            for i in range(len(valCorrect_list)):
                heads_val_acc[i] = valCorrect_list[i] / total_step_val
            avgHeadValAcc = sum(heads_val_acc) / len(heads_val_acc) # average val accuracy for all heads

            if avgHeadValAcc > best_acc:
                best_acc = avgHeadValAcc
                best_model = copy.deepcopy(model)
                best_optimizer = copy.deepcopy(optimizer)
        else:
            train_acc = trainCorrect / total_step_train
            val_acc = valCorrect / total_step_val

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(model)
                best_optimizer = copy.deepcopy(optimizer)

        # Logs
        if config["cnn"]["model"]["use_multihead"]:
            loss_dict["train_loss"].append(avgHeadTrainLoss)
            loss_dict["train_acc"].append(avgHeadTrainAcc)
            loss_dict["val_loss"].append(avgHeadValLoss)
            loss_dict["val_acc"].append(avgHeadValAcc)
        else:
            loss_dict["train_loss"].append(totalTrainLoss / total_step_train)
            loss_dict["train_acc"].append(train_acc)
            loss_dict["val_loss"].append(totalValLoss / total_step_val)
            loss_dict["val_acc"].append(val_acc)
    
        with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
            file.write(f"Epoch: {epoch+1}/{num_epochs}\n")
            file.write(f"\tTrain loss: {loss_dict['train_loss'][-1]:.4f}, Val loss: {loss_dict['val_loss'][-1]:.4f}\n")
            file.write(f"\tTrain accuracy: {loss_dict['train_acc'][-1]:.4f}, Val accuracy: {loss_dict['val_acc'][-1]:.4f}\n")
            if config["cnn"]["model"]["use_multihead"]:
                for i in range(len(heads_train_acc)):
                    file.write(f"\t\tHead {i+1}:\n")
                    file.write(f"\t\t\tTrain loss: {avgTrainLoss_heads[i]:.4f}, Val loss: {avgValLoss_heads[i]:.4f}\n")
                    file.write(f"\t\t\tTrain accuracy: {heads_train_acc[i]:.4f}, Val accuracy: {heads_val_acc[i]:.4f}\n")
            file.write(f"\tTime elapsed: {time.time() - start_time:.2f} seconds\n")

        if config["cnn"]["training"]["early_stopping"]:
            if avgHeadValLoss < best_val_loss:
                best_val_loss = avgHeadValLoss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print("Early stopping...")
                    break

    return best_model, best_optimizer

# ------------------------- MAIN -------------------------------
if __name__ == "__main__":
    config = json.load(open("config.json"))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If using a GPU, clear the cache:
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()

    print("Used device is: ", device)

    vis_utils = Visualization(config=config)
    prepro_utils = PreprocessingUtils(config=config)
    model_utils = ModelUtils(config=config, device=device)

    # define the active user and his data path
    user = config["active_user"]
    data_path = config["general"]["data_paths"][user]

    # create folders for logging
    log_folder_training, plot_folder_training, model_folder_training = create_folders_logging(config)

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
    num_workers = config["cnn"]["training"]["num_workers"]
    num_classes = config["cnn"]["model"]["num_classes"]

    transform = transforms.Compose([
        SimplePreprocessor(
            width=config["preprocessor"]["resize"]["width"],
            height=config["preprocessor"]["resize"]["height"]
        ),
        transforms.ToTensor()
    ])
    print("Loading dataset...")
    dataset = CustomDataset(data_path, transform=transform)

    if config["general"]["use_stratify"]:   
        # Get the labels from your dataset
        labels = [label for _, label in dataset]

        # Create a StratifiedShuffleSplit object
        sss = StratifiedShuffleSplit(n_splits=1, train_size=config["cnn"]["training"]["num_samples_subset"], random_state=config["cnn"]["training"]["seed"])

        # Get the subset indices
        print("Creating first subset...")
        subset_indices, _ = next(sss.split(range(len(dataset)), labels))

        # Create the subset
        subset = Subset(dataset, subset_indices)

        # Now split the subset into train, validation, and test sets
        print("Creating second subset...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=config["cnn"]["training"]["seed"])
        train_indices, test_indices = next(sss.split(subset_indices, [labels[i] for i in subset_indices]))

        # Split the remaining data into validation and test sets
        print("Creating third subset...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split / (1 - test_split), random_state=config["cnn"]["training"]["seed"])
        val_indices, test_indices = next(sss.split(test_indices, [labels[i] for i in test_indices]))

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_indices)

    else:
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

    if config["general"]["show_distribution"]:
        # Show the distribution of the classes in the subsets
        vis_utils.show_distribution(train_subset, val_subset, test_subset)

    # Normalize the data
    if config["cnn"]["training"]["normalization"]["use"]:
        print("Normalization started...")
        subsets = [train_subset, val_subset, test_subset]
        train_subset, val_subset, test_subset = prepro_utils.normalize_data(subsets, batch_size, shuffle, dataset, num_workers)

    # Create weighted random sampler
    if config["cnn"]["training"]["use_weighted_rnd_sampler"]:
        print("Creating weighted random sampler...")
        sampler, shuffle = prepro_utils.create_weighted_sampler(train_subset, num_classes)
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
        vis_utils.show_histogram([train_loader, val_loader, test_loader])

    # Choose and Initialize a model
    print("Initializing model...")
    model = model_utils.init_model(num_classes)

    # Define loss function and optimizer
    loss_functions = {
        "1": nn.CrossEntropyLoss(),
        "2": nn.MSELoss(),
        "3": nn.L1Loss(),
        "4": nn.NLLLoss()
    }
    optimizer = {
        "1": optim.AdamW(model.parameters(), lr=learning_rate),
        "2": optim.SGD(model.parameters(), lr=learning_rate)
    }

    if config["cnn"]["training"]["use_weighted_loss"]:
        print("Initializing weighted loss fcns...")
        print("\tComputing class weights...")
        class_weights = PreprocessingUtils.compute_class_weights(config, train_loader)
        criterion = []
        for i in range(len(class_weights)):
            class_weights[i] = class_weights[i].to(device)
            head_criterion = nn.CrossEntropyLoss(weight=class_weights[i])
            criterion.append(head_criterion)
    else:
        criterion = [loss_functions[config["cnn"]["training"]["loss_function"]].to(device) for _ in range(4)]

    optimizer = optimizer[config["cnn"]["training"]["optimizer"]]
    
    # Learning rate scheduling
    if config["cnn"]["training"]["lr_scheduler"]["use"]:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["cnn"]["training"]["lr_scheduler"]["step_size"], gamma=config["cnn"]["training"]["lr_scheduler"]["gamma"])

    # Load the model and optimizer if continue_training is set to True
    if config["cnn"]["training"]["continue_training"]:
        print("Loading model...")
        model_path = config["general"]["model_path_to_load"]
        model, optimizer = model_utils.load_model(model, optimizer, model_path)

    loss_dict = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Train the model
    print("Training started!")
    best_model, best_optimizer = train()
    # save the model
    if config["general"]["save_model"]:
        print("saving the model...")
        model_utils.save_model(best_model, best_optimizer, os.path.join(model_folder_training, "model.pth"))

    # Test the model
    print("Testing started...")
    if config["cnn"]["model"]["use_multihead"]:
        test_accuracy, heads_test_acc, test_f1_score = test_model(best_model, test_loader, device, config)
        with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
            file.write(f"\tTest accuracy: {test_accuracy * 100:.2f}%\n")
            file.write(f"\tTest F1 score: {test_f1_score * 100:.2f}\n")
            for i in range(len(heads_test_acc)):
                file.write(f"\t\tTest accuracy head {i+1}: {heads_test_acc[i] * 100:.2f}%\n")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    else:
        test_accuracy = test_model(best_model, test_loader, device)
        with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
            file.write(f"\tTest accuracy: {test_accuracy * 100:.2f}%")

    # Visualize the network
    if config["general"]["show_net_structure"]:    
        print("Visualizing the network...")
        model_graph = draw_graph(model, input_size=(batch_size, 128), device=device)
        model_graph.visual_graph

    # plot the training loss and accuracy
    print("Plotting learning curve")
    vis_utils.plot_learning_curve(loss_dict, plot_folder_training)

    print("All done!")