import os

import cv2
import numpy as np
import pandas as pd

from bisect import bisect_right
from PIL import Image
from tqdm import tqdm

class SimpleDataLoader():
    def __init__(self, data_path, preprocessors=None):
        self.csv_file_name = "caxton_dataset_filtered.csv"
        self.data_path = data_path
        if self.csv_file_name in os.listdir(self.data_path):
            self.data_frame = pd.read_csv(os.path.join(self.data_path, self.csv_file_name))
        else:
            self.data_frame = None
        self.num_samples_cumulated_per_folder = []
        self.num_samples = self.count_samples()
        
        self.current_idx = 0
        self.indices = np.arange(self.num_samples)
        np.random.shuffle(self.indices)

        self.preprocessors = preprocessors
        if self.preprocessors == None:
            self.preprocessors = []

    
    def __len__(self):

        return self.num_batches

    def count_samples(self):
        if self.data_frame is not None:
            num_samples = len(self.data_frame)
        else:
            num_samples = 0

        return num_samples

    def create_label(self, df_item):

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

    def load_data(self, num_samples):
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        indices = indices[:num_samples]

        data = []
        labels = []
        for idx in tqdm(indices, desc="Sample loading"):
            img_path = self.data_frame["img_path"][idx]
            img_path = os.path.join("./", img_path)

            # Load your data from file
            try:
                img = self.load_image(img_path)
                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        img = p.preprocess(img)
                data.append(img)
                labels.append(self.create_label(self.data_frame.iloc[idx]))
            except OSError as e:
                print("Error processing image:", e)
                continue

        return np.array(data), np.array(labels)
    
    def __iter__(self): # creates iterable object
        return self

    def __next__(self): # this method defines the iterator for the iterable object created by __iter__
        if self.current_idx >= self.num_samples:
            raise StopIteration

        sample_idx = self.indices[self.current_idx] 
        data = self.load_data(sample_idx)
        self.current_idx += 1
        
        return sample_idx, data
    