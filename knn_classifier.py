import os

import numpy as np
import cv2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from preprocessing.simple_preprocessor import SimplePreprocessor
from datasets.simple_dataloader import SimpleDataLoader

# demo implementation for knn, enter more details later
data_path = "path/to/your/dataset"
batch_size = 32
simple_preprocessor = SimplePreprocessor(width=32, height=32)
dataloader = SimpleDataLoader(data_path, batch_size=batch_size, preprocessors=simple_preprocessor)

knn = KNeighborsClassifier(n_neighbors=3)


for batch_data in dataloader:
    batch_images = []
    batch_labels = []
    
    for image_data, label in batch_data:
        batch_images.append(image_data)
        batch_labels.append(label)
    
    batch_images_flat = np.array(batch_images).reshape(len(batch_images), -1)
    batch_labels_flat = np.array(batch_labels)
    
    knn.partial_fit(batch_images_flat, batch_labels_flat, classes=np.unique(batch_labels_flat))


