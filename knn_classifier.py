import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score # for cross validation
from tqdm import tqdm

# Import our classes:
from preprocessing.simple_preprocessor import SimplePreprocessor
from datasets.simple_dataloader import SimpleDataLoader

# demo implementation for knn, enter more details later

data_path = "/Volumes/Samsung USB" # change this to your directory with dataset
simple_preprocessor = SimplePreprocessor(width=300, height=300) # width and height for the resizing
dataloader = SimpleDataLoader(data_path, preprocessors=simple_preprocessor)

data, labels = dataloader.load_data(num_samples_subset=3000)
# till here it works great :)
imgs_flat = data.reshape(data.shape[0], -1) # flatten the image matrix to 1D vector
labels_flat = labels.reshape(labels.shape[0], -1) # does this do anything ?

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
	k_accuracy.append(np.mean(accuracies)) # add the mean value of accuracies
print("End of cross validation.")

# print the accuracies:
plt.figure("KNN cross validation")
plt.plot(k_range, k_accuracy)
plt.show()

# test_accuracy = knn.score(testX, testY)
# print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
