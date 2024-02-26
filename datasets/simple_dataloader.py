import os

import cv2 as cv
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

        self.num_samples = self.count_samples()
        self.preprocessors = preprocessors

  # here i deleted the if preprocesors=None then preprocesors is []
  # because then in load_data() and count_samples() we are using "if self.preprocessors is not None:..."
  # which would be always true, ([] != None)
    
    def __len__(self):
        return self.num_samples

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
        image_data = cv.imread(image_path) # reading using openCV, its same as before but one line only.
                                           # returns RGB matrixes in np array
        return image_data

    def nozzle_coordinates(self, idx):
        # fcn to get coordinates of nozzle from dataset for specific image and save them to preprocessor atribute
        x = self.data_frame["nozzle_tip_x"][idx]
        y = self.data_frame["nozzle_tip_y"][idx]
        self.preprocessors.coordinates = [x, y]

    def load_data(self, num_samples_subset):
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        indices = indices[:num_samples_subset] # here we pick number of indices from shuffled indices

        data = []
        labels = []
        for idx in tqdm(indices, desc="Sample loading"):
            img_path = self.data_frame["img_path"][idx]
            img_path = os.path.join(self.data_path, img_path) # I changed this because we need to connect the path where is the dataset and the local path in dataset
            
            self.nozzle_coordinates(idx) # nozzle coordinates as [x, y] are saved to preprocesor.coordinates
            
            # Load your data from file
            try:
                img = self.load_image(img_path)
                
                if isinstance(self.preprocessors, list):  # Check if preprocessors is a list
                    for p in self.preprocessors:
                        img = p.preprocess(img)
                elif self.preprocessors is not None:  # Check if preprocessors is a single preprocessor instance
                    img = self.preprocessors.preprocess(img)

                data.append(img)
                labels.append(self.create_label(self.data_frame.iloc[idx]))
            except OSError as e:
                print("Error processing image:", e)
                continue

        return np.array(data), np.array(labels)