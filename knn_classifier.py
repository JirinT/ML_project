import json
import os
import time

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from preprocessing.simple_preprocessor import SimplePreprocessor
from datasets.simple_dataloader import SimpleDataLoader


config = json.load(open("config.json"))

# add your data_paths here
data_paths = {
	"jan": "./caxton_dataset",
	"leon": "test",
	"jiri": "/Volumes/Samsung USB",
	"remote_pc": "./caxton_dataset"
}

data_path = data_paths["jiri"] # change this to your name

plot_path = config["general"]["plot_path"]
logfile_path = config["general"]["log_path"]

simple_preprocessor = SimplePreprocessor(
	width=config["preprocessor"]["resize"]["width"], 
	height=config["preprocessor"]["resize"]["height"]
	)
dataloader = SimpleDataLoader(data_path, preprocessors=simple_preprocessor)

data, labels = dataloader.load_data(
	num_samples_subset=config["training"]["num_samples_subset"], 
	start_idx=config["training"]["start_idx"], 
	end_idx=config["training"]["end_idx"]
	)

imgs_flat = data.reshape(data.shape[0], -1) # flatten the image matrix to 1D vector
labels_flat = labels.reshape(labels.shape[0], -1) # flatten the labels matrix to 1D vector

show_images = config["general"]["show_sample_images"]
if show_images:
	for img_idx in range(len(data)):
		if img_idx > 5:
			break
		cv.imwrite(os.path.join(config["general"]["sample_img_path"], f"sample_image_{img_idx}.png"), data[img_idx])

(trainX, testX, trainY, testY) = train_test_split(
	imgs_flat, 
	labels_flat, 
	test_size=config["training"]["test_size"], 
	random_state=config["training"]["random_state"]
	) # stratify method throws error for me

if config["training"]["use_cross_validation"]:
	k_range = range(1,config["training"]["num_k"]) # k which will be tested
	k_accuracy = [] # here the accuracies for different k will be saved

	current_datetime = datetime.now()
	filename = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
	with open(os.path.join(logfile_path, f"log_{filename}.txt"), "w") as file:
		print("Cross validation started.")
		for k in k_range:
			knn = KNeighborsClassifier(n_neighbors=k, metric=config["classifier"]["distance_metric"]) # initialize the KNN with current k

			accuracies = cross_val_score(knn, X=trainX, y=trainY, cv=config["training"]["cv_fold"]) # returns 1D vector with the accuracies for each validation set, 
																		# cv is the number of folds used in cross validation
			file.write(f"K-value {k}: Accuracy = {np.mean(accuracies)}\n")
			k_accuracy.append(np.mean(accuracies))
	print("End of cross validation.")

	plt.figure("KNN cross validation")
	plt.plot(k_range, k_accuracy)
	plt.xlabel('Number of Neighbors (k)')
	plt.ylabel('Cross-Validation Accuracy')
	plt.title('KNN Cross-Validation Accuracy for Different k values')
	if config["general"]["save_cv_plot"]:
		plt.savefig(os.path.join(plot_path, "knn_cross_validation.png"))
	plt.show(block=False)
	time.sleep(5)
	plt.close()

	best_k = k_range[np.argmax(k_accuracy)]
	print(f'Best k based on cross-validation: {best_k}')

	knn = KNeighborsClassifier(n_neighbors=best_k, metric=config["classifier"]["distance_metric"])
	knn.fit(trainX, trainY)
else:
	k = config["classifier"]["k_value"]
	knn = KNeighborsClassifier(n_neighbors=k, metric=config["classifier"]["distance_metric"])
	knn.fit(trainX, trainY)

test_accuracy = knn.score(testX, testY)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
