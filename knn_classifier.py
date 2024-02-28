import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from preprocessing.simple_preprocessor import SimplePreprocessor
from datasets.simple_dataloader import SimpleDataLoader

# add your data_paths here
data_paths = {
	"jan": "./caxton_dataset",
	"leon": "test",
	"jiri": "test"
}
data_path = data_paths["jan"] # change this to your name

simple_preprocessor = SimplePreprocessor(width=30, height=30) # width and height for the resizing
dataloader = SimpleDataLoader(data_path, preprocessors=simple_preprocessor)

data, labels = dataloader.load_data(num_samples_subset=1000, start_idx=None, end_idx=None) #if you want you can specify a range of indices, that should be loadad

imgs_flat = data.reshape(data.shape[0], -1) # flatten the image matrix to 1D vector
labels_flat = labels.reshape(labels.shape[0], -1) # flatten the labels matrix to 1D vector

# # this is just for visualisation of preprocessed images:
# for img in data:
# 	cv.imshow("Sample image", img)
# 	cv.waitKey()

(trainX, testX, trainY, testY) = train_test_split(imgs_flat, labels_flat, test_size=0.25, 
stratify=labels_flat, random_state=42) # stratify method ensures that the labels will be distributed equally in train and test sets

# Cross validation:
k_range = range(1,50) # k which will be tested, we can try to increase the number based on observartions
k_accuracy = [] # here the accuracies for different k will be saved

print("Cross validation started.")
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k) # initialize the KNN with current k

	accuracies = cross_val_score(knn, X=trainX, y=trainY, cv=10) # returns 1D vector with the accuracies for each validation set, 
																 # cv is the number of folds used in cross validation
	k_accuracy.append(np.mean(accuracies))
print("End of cross validation.")

plt.figure("KNN cross validation")
plt.plot(k_range, k_accuracy)
plt.show()

# test_accuracy = knn.score(testX, testY)
# print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
