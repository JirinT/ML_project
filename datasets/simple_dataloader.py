import os

import cv2
import numpy as np
import pandas as pd

from bisect import bisect_right
from PIL import Image

class SimpleDataLoader():
    def __init__(self, data_path, batch_size, preprocessors=None):
        self.csv_file_name = "print_log_filtered"
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_samples_cumulated_per_folder = []
        self.num_samples = self.count_samples()
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.preprocessors = preprocessors

        if self.preprocessors == None:
            self.preprocessors = []
    
    def __len__(self):

        return self.num_batches

    def count_samples(self):
        num_samples = 0
        for folder_name in sorted(os.listdir(self.data_path)):
            folder_path = os.path.join(self.data_path, folder_name)
            if os.path.isdir(folder_path):
                if self.csv_file_name in os.listdir(folder_path):
                    df = pd.read_csv(os.path.join(folder_path, self.csv_file_name))
                    num_samples += len(df)
                    self.num_samples_cumulated_per_folder.append(num_samples)

        return num_samples

    def create_label(self, df_item):

        num_classes_per_label = [3, 3, 3, 3]
        label_flow_rate = df_item["flow_rate_class"]
        label_feed_rate = df_item["feed_rate_class"]
        label_z_offset = df_item["z_offset_class"]
        label_hotend = df_item["hotend_class"]
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
    
    def load_image(image_path):
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

    def load_data(self, idx):
        # Use binary search to find the folder containing the image with the current index
        folder_idx = bisect_right(self.num_samples_cumulated_per_folder, idx)
        if folder_idx > 0:
            file_idx = idx - self.num_samples_cumulated_per_folder[folder_idx - 1]
        else:
            file_idx = idx

        folder_name = sorted(os.listdir(self.data_path))[folder_idx]
        folder_path = os.path.join(self.data_path, folder_name)

        df = pd.read_csv(os.path.join(folder_path, self.csv_file_name))
        img_path = df["image_path"][file_idx]
        img_path = os.path.join("./", img_path)

        # Load your data from file
        img = self.load_image(img_path)
        if self.preprocessors is not None:
            for p in self.preprocessors:
                img = p.preprocess(img)
        data = img
        label = self.create_label(df[file_idx])

        return data, label
    
    def __iter__(self):
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_data = [self.load_data(idx) for idx in batch_indices]

            yield batch_data
