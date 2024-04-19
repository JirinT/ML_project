import json
import os
import torch

from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import transforms
import torch.nn as nn

from datasets.custom_dataset import CustomDataset
from preprocessing.simple_preprocessor import SimplePreprocessor

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

config = json.load(open("config.json"))

def one_hot_encode(tensor):
    # Find the index of the maximum value
    max_index = tensor.argmax()
    # Create a new tensor with zeros and a 1 at the max_index
    one_hot = torch.zeros_like(tensor)
    one_hot[max_index] = 1
    # one_hot = one_hot.numpy()

    return one_hot

def decode_labels(labels):
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

def decode_predictions(pred_heads_log_prob):
    prob_heads = [torch.exp(x) for x in pred_heads_log_prob]  # convert to standard probabilities
    pred_heads_list = []
    for j in range(len(prob_heads)):
        pred_head = torch.zeros((len(prob_heads[j]), 3))
        for i in range(len(prob_heads[j])):
            pred_head[i] = one_hot_encode(prob_heads[j][i])
        pred_heads_list.append(pred_head)

    return pred_heads_list

def conf_matrix(outputs, labels):
    y_true = [[] for _ in range(config["cnn"]["model"]["num_heads"])]
    y_pred = [[] for _ in range(config["cnn"]["model"]["num_heads"])]

    decoded_labels = decode_labels(labels.cpu().numpy()) # makes a list of 4 arrays of shape (batch_size, 1)

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
    conf_matrix_path = config["general"]["conf_matrix_path"]
    os.makedirs(conf_matrix_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'confusion_matrices_{timestamp}.png'

    plt.savefig(os.path.join(conf_matrix_path, filename))

def test_model(model, test_loader, device, config):
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
    heads_test_acc = [0, 0, 0, 0]
    correct = 0
    testCorrect_total = 0
    pred_labels = [torch.empty((0,3)).to(device) for _ in range(config["cnn"]["model"]["num_heads"])]
    true_labels = torch.empty((0,12)).to(device)
    model.eval()
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.to(device)
            labels = labels.to(device) # labels is a tensor of shape (batch_size x 12)

            if config["cnn"]["model"]["type"]["multihead"]:
                x1, x2, x3, x4 = model(images) # x1, x2, x3, x4 are outputs of last linear layer - raw data
                raw_predictions = [x1, x2, x3, x4] # raw_predictions are outputs of last linear layer, before LogSoftMax

                for i, x in enumerate(raw_predictions):
                    pred_labels[i] = torch.cat((pred_labels[i], x), dim=0)
                true_labels = torch.cat((true_labels, labels), dim=0)

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
                heads_test_acc[i] = correct_list[i] / len(test_loader.dataset)
            total_accuracy = testCorrect_total / len(test_loader.dataset)

            if config["general"]["log_confusion_matrix"]:
                print("Creating confusion matrix...")
                conf_matrix(pred_labels, true_labels)

            return total_accuracy, heads_test_acc
        else:
            accuracy = correct / len(test_loader.dataset)
            return accuracy

if __name__ == "__main__":
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
    print("Loading the dataset, loading model, creating data loaders...")
    dataset = CustomDataset(data_path, transform=transform)

    num_samples_train = int(train_split * len(dataset))
    num_samples_val = int(val_split * len(dataset))
    num_samples_test = int(test_split * len(dataset))

    (_, _, test_set) = random_split(dataset, [num_samples_train, num_samples_val, num_samples_test], generator=torch.Generator().manual_seed(config["cnn"]["training"]["seed"]))

    num_samples_test_subset = int(config["cnn"]["training"]["num_samples_subset"] * test_split)

    test_subset = Subset(test_set, range(num_samples_test_subset))

    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=shuffle)

    # model = torch.load(os.path.join(config["general"]["model_path"], "model.pth")).to(device)
    model = torch.load("./logs/models/2024-04-14_19-14-33/model.pth").to(device)

    print("Testing the model...")
    accuracy,_ = test_model(model, test_loader, device, config)
    print(f'Accuracy: {accuracy * 100:.2f}%')