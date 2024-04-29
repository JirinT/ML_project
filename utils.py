import os
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import transforms


from cnn import CNN2, CNN4
from MultiHeadNetwork import MultiHeadNetwork
from preprocessing.simple_preprocessor import SimplePreprocessor

class Visualization():
    def __init__(self, config):
        self.config = config
        self.prepro_utils = PreprocessingUtils(config)

    def plot_learning_curve(self, loss_dict, plot_folder_training):
        fig, ax1 = plt.subplots()
        # Plot train and validation acc
        accuracy_train, = ax1.plot(loss_dict["train_acc"], 'b-', label='Train Accuracy')
        accuracy_val, = ax1.plot(loss_dict["val_acc"], 'b--', label='Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy', color='b')
        ax1.tick_params(axis='y', colors='b')

        # Create second axis (for loss)
        ax2 = ax1.twinx()

        # Plot train and validation loss on the second axis
        loss_train, = ax2.plot(loss_dict["train_loss"], 'r-', label='Train Loss')
        loss_val, = ax2.plot(loss_dict["val_loss"], 'r--', label='Validation Loss')
        ax2.set_ylabel('Loss', color='r')
        ax2.tick_params(axis='y', colors='r')

        lines = [accuracy_train, accuracy_val, loss_train, loss_val]
        labels = ['Train Accuracy', 'Validation Accuracy', 'Train Loss', 'Validation Loss']
        plt.legend(lines, labels, loc='upper left')

        plt.savefig(os.path.join(plot_folder_training, "loss_plot.svg"), format='svg')
    
    def show_histogram(self, loaders):
        if self.config["cnn"]["model"]["use_multihead"]:
            for loader in loaders:
                counts_flow_rate = torch.zeros(3)
                counts_lateral_speed = torch.zeros(3)
                counts_z_offset = torch.zeros(3)
                counts_hotend_temperature = torch.zeros(3)
                for _, labels in loader:
                    flow_rate_labels = labels[:, 0:3]
                    lateral_speed_labels = labels[:, 3:6]
                    z_offset_labels = labels[:, 6:9]
                    hotend_temperature_labels = labels[:, 9:]
                    counts_flow_rate += flow_rate_labels.sum(axis=0)
                    counts_lateral_speed += lateral_speed_labels.sum(axis=0)
                    counts_z_offset += z_offset_labels.sum(axis=0)
                    counts_hotend_temperature += hotend_temperature_labels.sum(axis=0)
                
                plt.figure("Histograms for each label", figsize=(13, 8))
                plt.subplot(2, 2, 1)
                plt.bar(range(len(counts_flow_rate)), counts_flow_rate)
                plt.xticks([0, 1, 2], ['Low', 'Good', 'High'])
                plt.ylabel("Amount of samples")
                plt.title("Flow Rate")

                plt.subplot(2, 2, 2)
                plt.bar(range(len(counts_lateral_speed)), counts_lateral_speed)
                plt.xticks([0, 1, 2], ['Low', 'Good', 'High'])
                plt.ylabel("Amount of samples")
                plt.title("Lateral Speed")

                plt.subplot(2, 2, 3)
                plt.bar(range(len(counts_z_offset)), counts_z_offset)
                plt.xticks([0, 1, 2], ['Low', 'Good', 'High'])
                plt.ylabel("Amount of samples")
                plt.title("Z Offset")

                plt.subplot(2, 2, 4)
                plt.bar(range(len(counts_hotend_temperature)), counts_hotend_temperature)
                plt.xticks([0, 1, 2], ['Low', 'Good', 'High'])
                plt.ylabel("Amount of samples")
                plt.title("Hotend Temperature")
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(os.path.join(self.config["general"]["histogram_path"], f"histogram_multihead_{timestamp}.svg"), format="svg")

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
            plt.savefig(os.path.join(self.config["general"]["histogram_path"], f"histogram_separate_{type}_{timestamp}.png"))

    def show_distribution(self, train_subset, val_subset, test_subset):
        target_dict = defaultdict(int) 
        for _, label in train_subset:
            label_indices = tuple(label.tolist())  # Convert one-hot encoded labels to class indices and create a tuple
            target_dict[label_indices] += 1
        
        target_dict = {key: target_dict[key] for key in sorted(target_dict)}
        total = sum(target_dict.values())
        target_dict = {key: value/total for key, value in target_dict.items()}
        plt.figure()
        keys_str = [str(key) for key in target_dict.keys()]
        plt.bar(keys_str, target_dict.values(), color='skyblue')
        plt.xticks(range(len(keys_str)), range(1, len(keys_str) + 1))
        plt.title("Distribution of classes in the TRAINING subset")

        target_dict = defaultdict(int) 
        for _, label in val_subset:
            label_indices = tuple(label.tolist())  # Convert one-hot encoded labels to class indices and create a tuple
            target_dict[label_indices] += 1
        
        target_dict = {key: target_dict[key] for key in sorted(target_dict)}
        total = sum(target_dict.values())
        target_dict = {key: value/total for key, value in target_dict.items()}
        plt.figure()
        keys_str = [str(key) for key in target_dict.keys()]
        plt.bar(keys_str, target_dict.values(), color='skyblue')
        plt.xticks(range(len(keys_str)), range(1, len(keys_str) + 1))
        plt.title("Distribution of classes in the VALIDATION subset")

        target_dict = defaultdict(int)
        for _, label in test_subset:
            label_indices = tuple(label.tolist())
            target_dict[label_indices] += 1

        target_dict = {key: target_dict[key] for key in sorted(target_dict)}
        total = sum(target_dict.values())
        target_dict = {key: value/total for key, value in target_dict.items()}
        plt.figure()
        keys_str = [str(key) for key in target_dict.keys()]
        plt.bar(keys_str, target_dict.values(), color='skyblue')
        plt.xticks(range(len(keys_str)), range(1, len(keys_str) + 1))
        plt.title("Distribution of classes in the TEST subset")
        plt.show()

    def conf_matrix(self, outputs, labels):
        y_true = [[] for _ in range(self.config["cnn"]["model"]["num_heads"])]
        y_pred = [[] for _ in range(self.config["cnn"]["model"]["num_heads"])]

        decoded_labels = self.prepro_utils.decode_labels(labels.cpu().numpy()) # makes a list of 4 arrays of shape (batch_size, 1)

        for i, output in enumerate(outputs):
                _, predicted = torch.max(output, 1)
                y_pred[i].extend(predicted.tolist())
                y_true[i].extend(decoded_labels[i].tolist())

        # Create a subplot for each head
        _, axs = plt.subplots(2, 2, figsize=(8, 5))
        titles = ['Flow Rate', 'Lateral Speed', 'Z offset', 'Hotend Temperature']
        # Create a confusion matrix for each output head
        for i, ax in enumerate(axs.flatten()):
            cm = confusion_matrix(y_true[i], y_pred[i], labels=[0, 1, 2])
            cmd = ConfusionMatrixDisplay(cm, display_labels=["Low", "Good", "High"])

            # Plot the confusion matrix
            cmd.plot(ax=ax)
            ax.set_title(titles[i])
        plt.tight_layout()

        # Save the plot
        conf_matrix_path = self.config["general"]["conf_matrix_path"]
        os.makedirs(conf_matrix_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'confusion_matrices_{timestamp}.png'

        plt.savefig(os.path.join(conf_matrix_path, filename))

class PreprocessingUtils():
    def __init__(self, config):
        self.config = config

    def get_mean_std(self, loader):
        num_pixels = 0
        mean = 0.0
        std = 0.0
        for images, _ in loader:
            batch_size, num_channels, height, width = images.shape
            num_pixels += batch_size * height * width
            mean += images.mean(axis=(0, 2, 3)).sum()
            std += images.std(axis=(0, 2, 3)).sum()

        mean /= num_pixels
        std /= num_pixels

        return mean, std

    def normalize_data(self, subsets, batch_size, shuffle, dataset, num_workers):
        train_subset = subsets[0]
        val_subset = subsets[1]
        test_subset = subsets[2]

        log_dir = self.config["cnn"]["training"]["normalization"]["log_path"]
        file_name = "mean_std.json"
        file_path = os.path.join(log_dir, file_name)

        train_loader_for_normalization = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.custom_collate_fn, num_workers=num_workers)
        if self.config["cnn"]["training"]["normalization"]["compute_new_mean_std"] or not os.path.exists(file_path):
            mean, std = self.get_mean_std(train_loader_for_normalization)

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
            width=self.config["preprocessor"]["resize"]["width"], 
            height=self.config["preprocessor"]["resize"]["height"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std) # normalize the images
        ])
        # Apply the transform to the datasets
        train_subset.dataset.transform = transform
        val_subset.dataset.transform = transform
        test_subset.dataset.transform = transform

        return train_subset, val_subset, test_subset

    def create_weighted_sampler(self, train_subset, num_classes):
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
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False

        return sampler, shuffle
    
    def decode_labels(self, labels):
        flow_rate = np.zeros((3,))
        lateral_speed = np.zeros((3,))
        z_offset = np.zeros((3,))
        hotend_temperature = np.zeros((3,))

        for i in range(labels.shape[0]):
            flow_rate = np.vstack((flow_rate,labels[i][:3]))
            lateral_speed = np.vstack((lateral_speed,labels[i][3:6]))
            z_offset = np.vstack((z_offset,labels[i][6:9]))
            hotend_temperature = np.vstack((hotend_temperature,labels[i][9:]))

        flow_rate_decoded = np.argmax(flow_rate, axis=1)
        lateral_speed_decoded = np.argmax(lateral_speed, axis=1)
        z_offset_decoded = np.argmax(z_offset, axis=1)
        hotend_temperature_decoded = np.argmax(hotend_temperature, axis=1)

        flow_rate_decoded = flow_rate_decoded[1:]
        lateral_speed_decoded = lateral_speed_decoded[1:]
        z_offset_decoded = z_offset_decoded[1:]
        hotend_temperature_decoded = hotend_temperature_decoded[1:]

        return [flow_rate_decoded, lateral_speed_decoded, z_offset_decoded, hotend_temperature_decoded]
    
    def compute_class_weights(config, data_loader):
        if config["cnn"]["model"]["use_multihead"]:
            counts_flow_rate = torch.zeros(3)
            counts_lateral_speed = torch.zeros(3)
            counts_z_offset = torch.zeros(3)
            counts_hotend_temperature = torch.zeros(3)

            for _, labels in data_loader:
                flow_rate_labels = labels[:, 0:3]
                lateral_speed_labels = labels[:, 3:6]
                z_offset_labels = labels[:, 6:9]
                hotend_temperature_labels = labels[:, 9:]
                counts_flow_rate += flow_rate_labels.sum(axis=0)
                counts_lateral_speed += lateral_speed_labels.sum(axis=0)
                counts_z_offset += z_offset_labels.sum(axis=0)
                counts_hotend_temperature += hotend_temperature_labels.sum(axis=0)
        else:
            return None
        # compute weights:
        N = len(data_loader.dataset)
        num_classes = config["cnn"]["model"]["num_classes"]
        class_weights = [0.5 * N / counts * num_classes for counts in [counts_flow_rate, counts_lateral_speed, counts_z_offset, counts_hotend_temperature]]
        return class_weights

class ModelUtils():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.prepro_utils = PreprocessingUtils(config)

    def init_model(self, num_classes):
        if self.config["cnn"]["model"]["type"]["cnn2"]:
            model = CNN2(config=self.config).to(self.device)

        elif self.config["cnn"]["model"]["type"]["cnn4"]:
            model = CNN4(config=self.config).to(self.device)

        elif self.config["cnn"]["model"]["type"]["resnet18"]:
            model = models.resnet18(weights=None, num_classes=num_classes).to(self.device)

        elif self.config["cnn"]["model"]["type"]["resnet34"]:
            model = models.resnet34(weights=None, num_classes=num_classes).to(self.device) 
        
        elif self.config["cnn"]["model"]["type"]["resnet50"]:
            model = models.resnet50(weights=None, num_classes=num_classes).to(self.device)
        
        else:
            raise ValueError("Model type not supported!")

        if self.config["cnn"]["model"]["use_multihead"]:
            shared_backbone = model
            # feature extraction - getting the output layer after convolution layers:
            if self.config["cnn"]["model"]["type"]["resnet18"] or self.config["cnn"]["model"]["type"]["resnet34"] or self.config["cnn"]["model"]["type"]["resnet50"]:
                return_nodes = {
                    "avgpool": "AdaptiveAvgPool2d(output_size=(1, 1))"
                }
            elif self.config["cnn"]["model"]["type"]["cnn2"]:
                return_nodes = {
                    "flatten": "Flatten(start_dim=1, end_dim=-1)"
                }
            elif self.config["cnn"]["model"]["type"]["cnn4"]:
                return_nodes = {
                    "flatten": "Flatten(start_dim=1, end_dim=-1)"
                }
            backbone_last_layer = create_feature_extractor(shared_backbone, return_nodes=return_nodes) # the backbone_last_layer is the output of the last convolutional layer which we feed into each head
            model = MultiHeadNetwork(self.config, backbone_last_layer).to(self.device)

        return model
    
    def save_model(self, model, optimizer, path):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, path)

    def load_model(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

            return model, optimizer
        else:
            return model
        
    def apply_regularization(self, model, loss):
        lambda_regularization = self.config["cnn"]["model"]["regularization"]["lambda"]
        if self.config["cnn"]["model"]["regularization"]["lasso"]:
            p = 1
        elif self.config["cnn"]["model"]["regularization"]["ridge"]:
            p = 2
        # L1 regularization term - LASSO
        l1 = torch.tensor(0.)
        l1 = l1.to(self.device)
        for param in model.parameters():
            l1 += torch.norm(param, p=p)

        loss += lambda_regularization * l1
        
        return loss
    
    def one_hot_encode(self, tensor):
        # Find the index of the maximum value
        max_index = tensor.argmax()
        # Create a new tensor with zeros and a 1 at the max_index
        one_hot = torch.zeros_like(tensor)
        one_hot[max_index] = 1

        return one_hot

    def decode_predictions(self, pred_heads_log_prob):
        prob_heads = [torch.exp(x) for x in pred_heads_log_prob]  # convert to standard probabilities
        pred_heads_list = []
        for j in range(len(prob_heads)):
            pred_head = torch.zeros((len(prob_heads[j]), 3))
            for i in range(len(prob_heads[j])):
                pred_head[i] = self.one_hot_encode(prob_heads[j][i])
            pred_heads_list.append(pred_head)

        return pred_heads_list
            
    def compute_losses(self, model, images, labels, config, criterion):
        x1, x2, x3, x4 = model(images) # x1, x2, x3, x4 are outputs of last linear layer - raw data
        raw_predictions = [x1, x2, x3, x4] # raw_predictions are outputs of last linear layer, before LogSoftMax

        pred_heads_log_prob = [nn.LogSoftmax(dim=1)(x) for x in raw_predictions] # pred_heads_log_prob are the log probabilities
        pred_heads = self.decode_predictions(pred_heads_log_prob) # pred_heads is now a list of 4 tensors of shape (batch_size x 3) and contains [0,1,0] for example
        pred_heads = [x.to(self.device) for x in pred_heads]

        # Compute the loss for each head and update the head parameters:
        if config["cnn"]["training"]["loss_function"] != "1":
            losses = [criterion[i](x, labels[:,i*3:(i+1)*3]) for i, x in enumerate([x1, x2, x3, x4])]
        else:
            losses = [criterion[i](x, labels[:,i*3:(i+1)*3].argmax(1)) for i, x in enumerate([x1, x2, x3, x4])]
        total_loss = sum(losses)

        return total_loss, losses, pred_heads
    
    def compute_f1_score(self, outputs, labels):
        y_true = [[] for _ in range(self.config["cnn"]["model"]["num_heads"])]
        y_pred = [[] for _ in range(self.config["cnn"]["model"]["num_heads"])]

        decoded_labels = self.prepro_utils.decode_labels(labels.cpu().numpy()) # makes a list of 4 arrays of shape (batch_size, 1)

        for i, output in enumerate(outputs):
                _, predicted = torch.max(output, 1)
                y_pred[i].extend(predicted.tolist())
                y_true[i].extend(decoded_labels[i].tolist())

        f1_score_head1 = f1_score(y_true[0], y_pred[0], average="micro")
        f1_score_head2 = f1_score(y_true[1], y_pred[1], average="micro")
        f1_score_head3 = f1_score(y_true[2], y_pred[2], average="micro")
        f1_score_head4 = f1_score(y_true[3], y_pred[3], average="micro")

        return f1_score_head1, f1_score_head2, f1_score_head3, f1_score_head4