import os

import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from PIL import Image
from torch.utils.data import Dataset


config = json.load(open("config.json"))

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.csv_file_name = config["general"]["csv_file_name"]
        self.data_path = data_path

        if self.csv_file_name in os.listdir(self.data_path):
            self.data_frame = pd.read_csv(os.path.join(self.data_path, self.csv_file_name))
        else:
            self.data_frame = None

        self.num_samples = self.count_samples()
        self.transform = transform
        self.preprocessors = transform.transforms[0]

    def __len__(self):
        return self.num_samples

    def count_samples(self):
        """
        Count the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        if self.data_frame is not None:
            num_samples = len(self.data_frame)
        else:
            num_samples = 0

        return num_samples

    def create_label(self, df_item):
        """
        Create a one-hot encoded label vector for a single sample.
        
        Args:
            df_item (pandas.Series): A single row from the dataset DataFrame.

        Returns:
            numpy.ndarray: The one-hot encoded label vector.
        """
        num_classes_per_label = [3, 3, 3, 3]
        label_flow_rate = df_item.loc["flow_rate_class"]
        label_feed_rate = df_item.loc["feed_rate_class"]
        label_z_offset = df_item.loc["z_offset_class"]
        label_hotend = df_item.loc["hotend_class"]
        labels = [label_flow_rate, label_feed_rate, label_z_offset, label_hotend]

        label_vector = self.one_hot_encoding_multilabel(labels, num_classes_per_label)

        return label_vector

    def one_hot_encoding_multilabel(self, labels, num_classes_per_label):
        """
        Perform one-hot encoding for multiple labels.
        
        Args:
            labels (list): List of label values for each label.
            num_classes_per_label (list): List containing the total number of classes for each label.
            
        Returns:
            numpy.ndarray: The concatenated one-hot encoded vector.
        """
        one_hot_vectors = []
        for label, num_classes in zip(labels, num_classes_per_label):
            one_hot_vector = np.zeros(num_classes)
            one_hot_vector[label] = 1
            one_hot_vectors.append(one_hot_vector)

        return np.concatenate(one_hot_vectors)
    
    def load_image(self, image_path):
        """
        Load image data from file.
        
        Args:
            file_path (str): The path to the image file.
            
        Returns:
            numpy.ndarray: The image data as a NumPy array.
        """
        image = Image.open(image_path)
        image_data = np.array(image)

        return image_data

    def nozzle_coordinates(self, idx):
        """
        get coordinates of nozzle from dataset for specific image and save them to preprocessor atribute

        Args:
            idx (int): index of the image in dataset
       
        """
        x = self.data_frame["nozzle_tip_x"][idx]
        y = self.data_frame["nozzle_tip_y"][idx]
        self.preprocessors.coordinates = [x, y]
    
    def __getitem__(self, index):
        """
        get item from dataset at given index

        Args:
            index (int): index of the item in dataset

        """
        img_path = self.data_frame["img_path"][index]

        self.nozzle_coordinates(index) # nozzle coordinates as [x, y] are saved to preprocesor.coordinates

        try:
            img = self.load_image(img_path)
            label = self.create_label(self.data_frame.iloc[index])
            
            if self.transform:
                img = self.transform(img)
                label = torch.tensor(label, dtype=torch.float32)
                label = label[:3]

        except Exception as e:
            print("Error processing image:", e)
            return None, None
        
        return img, label
    

    def custom_collate_fn(self, batch):
        """
        Custom collate function for the data loader.

        Args:
            batch (list): The batch to collate.

        Returns:
            tuple: The collated batch.
        """
        # Filter out samples where either the image or label is None
        batch = [sample for sample in batch if sample[0] is not None and sample[1] is not None]

        return data.default_collate(batch)