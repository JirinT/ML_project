import os

import numpy as np
import cv2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocessing.simple_preprocessor import SimplePreprocessor
from datasets.simple_dataloader import SimpleDataLoader

# demo implementation for knn, enter more details later

data_path = "/Volumes/Samsung USB" # change this to your directory with dataset
simple_preprocessor = SimplePreprocessor(width=32, height=32)
dataloader = SimpleDataLoader(data_path, preprocessors=simple_preprocessor)

data, labels = dataloader.load_data(num_samples_subset=10)
# till here it works great :)
imgs_flat = data.reshape(data.shape[0], -1)
labels_flat = labels.reshape(labels.shape[0], -1)

(trainX, testX, trainY, testY) = train_test_split(imgs_flat, labels_flat,
	test_size=0.25, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(trainX, trainY)

test_accuracy = knn.score(testX, testY)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
