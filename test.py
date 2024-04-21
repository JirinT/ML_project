import json

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import transforms

from datasets.custom_dataset import CustomDataset
from preprocessing.simple_preprocessor import SimplePreprocessor
from utils import Visualization, PreprocessingUtils, ModelUtils


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
    vis_utils = Visualization(config=config)
    prepro_utils = PreprocessingUtils(config=config)
    model_utils = ModelUtils(config=config, device=device)

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

            if config["cnn"]["model"]["use_multihead"]:
                x1, x2, x3, x4 = model(images) # x1, x2, x3, x4 are outputs of last linear layer - raw data
                raw_predictions = [x1, x2, x3, x4] # raw_predictions are outputs of last linear layer, before LogSoftMax

                for i, x in enumerate(raw_predictions):
                    pred_labels[i] = torch.cat((pred_labels[i], x), dim=0)
                true_labels = torch.cat((true_labels, labels), dim=0)

                pred_heads_log_prob = [nn.LogSoftmax(dim=1)(x) for x in raw_predictions] # pred_heads_log_prob are the log probabilities
                pred_heads = model_utils.decode_predictions(pred_heads_log_prob) # pred_heads is now a list of 4 tensors of shape (batch_size x 3) and contains [0,1,0] for example
                pred_heads = [x.to(device) for x in pred_heads]

                for i in range(len(pred_heads)):
                    correct_list[i] += (pred_heads[i].argmax(1) == labels[:,i*3:(i+1)*3].argmax(1)).type(torch.float).sum().item()
                    
                comparison = torch.all(torch.cat(pred_heads, dim=1) == labels, dim=1)
                testCorrect_total += torch.sum(comparison).item()
            else:
                pred = model(images)
                correct += (pred.argmax(1) == labels.argmax(1)).type(
                torch.float).sum().item()
        
        if config["cnn"]["model"]["use_multihead"]:
            for i in range(len(correct_list)):
                heads_test_acc[i] = correct_list[i] / len(test_loader.dataset)
            total_accuracy = testCorrect_total / len(test_loader.dataset)
            avgHeadAcc = sum(heads_test_acc) / len(heads_test_acc)

            f1_score_head1, f1_score_head2, f1_score_head3, f1_score_head4 = model_utils.compute_f1_score(pred_labels, true_labels)
            avg_f1_score = (f1_score_head1 + f1_score_head2 + f1_score_head3 + f1_score_head4) / 4

            if config["general"]["log_confusion_matrix"]:
                print("Creating confusion matrix...")
                vis_utils.conf_matrix(pred_labels, true_labels)

            return avgHeadAcc, heads_test_acc, avg_f1_score
        else:
            accuracy = correct / len(test_loader.dataset)

            return accuracy

if __name__ == "__main__":
    config = json.load(open("config.json"))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vis_utils = Visualization(config=config)
    prepro_utils = PreprocessingUtils(config=config)
    model_utils = ModelUtils(config=config, device=device)

    # define the active user and his data path
    user = config["active_user"]
    data_path = config["general"]["data_paths"][user]

    # Define hyperparameters
    batch_size = config["cnn"]["training"]["batch_size"]
    shuffle = config["cnn"]["training"]["shuffle"]
    train_split = config["cnn"]["training"]["train_split"]
    val_split = config["cnn"]["training"]["val_split"]
    test_split = config["cnn"]["training"]["test_split"]
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
    print("Loading the dataset, loading model, creating data loaders...")
    dataset = CustomDataset(data_path, transform=transform)

    num_samples_train = int(train_split * len(dataset))
    num_samples_val = int(val_split * len(dataset))
    num_samples_test = int(test_split * len(dataset))

    (_, _, test_set) = random_split(dataset, [num_samples_train, num_samples_val, num_samples_test], generator=torch.Generator().manual_seed(config["cnn"]["training"]["seed"]))

    num_samples_test_subset = int(config["cnn"]["training"]["num_samples_subset"] * test_split)

    test_subset = Subset(test_set, range(num_samples_test_subset))

    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.custom_collate_fn, num_workers=num_workers)

    # Load the model
    model = model_utils.init_model(num_classes)

    model_path = "./logs/models/2024-04-21_12-58-04/model.pth"
    model = model_utils.load_model(model, optimizer=None, path=model_path)

    print("Testing the model...")
    accuracy,_, f1_score = test_model(model, test_loader, device, config)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'F1 Score: {f1_score}')