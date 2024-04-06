import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torchvision.transforms import transforms
from datasets.custom_dataset import CustomDataset
from preprocessing.simple_preprocessor import SimplePreprocessor
from PIL import Image
import os
import numpy as np

def create_directory_structure(root_dir, dataset_splits):

    for dataset, subset in dataset_splits:
        subset_dir = os.path.join(root_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        print(f"Creating directory for {subset} set...")
        for idx, sample in enumerate(dataset):
            img = sample[0]
            label = sample[1].numpy()
            if np.array_equal(np.array([1, 0, 0]), label):
                class_name = "low"
            elif np.array_equal(np.array([0, 1, 0]), label):
                class_name = "good"
            else:
                class_name = "high"
            class_dir = os.path.join(subset_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            image = Image.fromarray(img)
            img_filename = f"image_{idx}.jpg"
            image.save(os.path.join(class_dir, img_filename))

config = json.load(open("config.json"))

# define the active user and his data path
user = config["active_user"]
data_path = config["general"]["data_paths"][user]

# Define split parameters
train_split = config["cnn"]["training"]["train_split"]
val_split = config["cnn"]["training"]["val_split"]
test_split = config["cnn"]["training"]["test_split"]

# Initialize dataset and data loader
transform = transforms.Compose([
    SimplePreprocessor(
    width=config["preprocessor"]["resize"]["width"], 
    height=config["preprocessor"]["resize"]["height"]
    )
])

# Initialize dataset
dataset = CustomDataset(data_path, transform=transform)

num_samples_train = int(train_split * len(dataset))
num_samples_val = int(val_split * len(dataset))
num_samples_test = int(test_split * len(dataset))

(train_set, val_set, test_set) = random_split(dataset, [num_samples_train, num_samples_val, num_samples_test], generator=torch.Generator().manual_seed(config["cnn"]["training"]["seed"]))

root_directory = config["general"]["data_paths"]["resnet"][user]

dataset_splits = [
    (train_set, "train"),
    (val_set, "val"),
    (test_set, "test")
]

create_directory_structure(root_directory, dataset_splits)