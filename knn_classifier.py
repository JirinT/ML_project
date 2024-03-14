import json
import os
import time

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import folder_functions

from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from preprocessing.simple_preprocessor import SimplePreprocessor
from datasets.simple_dataloader import SimpleDataLoader


config = json.load(open("config.json"))

now = datetime.now()
now_formated = now.strftime("%Y-%m-%d_%H-%M-%S")
log_folder_training = os.path.join(config["general"]["log_path"], now_formated)
os.makedirs(log_folder_training, exist_ok=True)
plot_folder_training = os.path.join(config["general"]["plot_path"], now_formated)
os.makedirs(plot_folder_training, exist_ok=True)

# Save the config to a text file

filename_config = os.path.join(log_folder_training, "config.txt")
with open(filename_config, 'w') as f:
    json.dump(config, f)

user = config["active_user"] # Change your name in config file (jan, leon, jiri, remote_pc)
data_path = config["general"]["data_paths"][user]

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
folder_functions.create_folder(config["general"]["sample_img_path"]) # creates the folder if it does not exist yet
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
	)

if config["training"]["use_cross_validation"]:
	k_range = range(1,config["training"]["num_k"]) # k which will be tested
	k_accuracy = [] # here the accuracies for different k will be saved

	with open(os.path.join(log_folder_training, "log.txt"), "w") as file:
		print("Cross validation started.")
		for k in k_range:
			print(f"Now running for k = {k}")
			knn = KNeighborsClassifier(n_neighbors=k, metric=config["classifier"]["distance_metric"]) # initialize the KNN with current k

			accuracies = cross_val_score(knn, X=trainX, y=trainY, cv=config["training"]["cv_fold"]) # returns 1D vector with the accuracies for each validation set, 
																		# cv is the number of folds used in cross validation
			file.write(f"K-value {k}: Accuracy = {np.mean(accuracies)}\n")
			k_accuracy.append(np.mean(accuracies))
	print("End of cross validation.")

	plt.figure("KNN cross validation")
	plt.plot(k_range, k_accuracy, c="b")
	plt.scatter(k_range,k_accuracy, marker=".", c="b", s=100)
	plt.xlabel('Number of Neighbors (k)')
	plt.ylabel('Cross-Validation Accuracy')
	plt.grid(True)
	plt.title('KNN Cross-Validation Accuracy for Different k values')

	if config["general"]["save_cv_plot"]:
		plt.savefig(os.path.join(plot_folder_training, "knn_cross_validation.png"))
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

predicted_labels = knn.predict(testX)

correct_classification = np.zeros(shape=(testX.shape[1],)) # here the correctly classified images will be stored
incorrect_classification = np.zeros(shape=(testX.shape[1],)) # here the incorrectly classified images will be stored

for i in range(testY.shape[0]):
	comparison = np.array_equal(testY[i], predicted_labels[i])
	if comparison == True:
		correct_classification = np.hstack((correct_classification,testX[i]))
	else:
		incorrect_classification = np.hstack((incorrect_classification,testX[i]))

correct_classification = correct_classification.reshape(-1, config["preprocessor"]["resize"]["height"], config["preprocessor"]["resize"]["width"])
incorrect_classification = incorrect_classification.reshape(-1, config["preprocessor"]["resize"]["height"], config["preprocessor"]["resize"]["width"])

# Save the paths for saving images:
correct_class_path = config["general"]["classified_images_path"]["correct"]
incorrect_class_path = config["general"]["classified_images_path"]["incorrect"]

# create the folders if they dont exist yet:
folder_functions.create_folder(correct_class_path)
folder_functions.create_folder(incorrect_class_path)

# Delete the current files in the folders:
folder_functions.delete_files(correct_class_path)
folder_functions.delete_files(incorrect_class_path)

# Save the images:
folder_functions.save_images(correct_class_path, correct_classification)
folder_functions.save_images(incorrect_class_path, incorrect_classification)

# Confusion matrix:
# conf_matrix = confusion_matrix(testY, predicted_labels) # this gives an error, needs preproces testY and predicted_labels

test_accuracy = knn.score(testX, testY)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
