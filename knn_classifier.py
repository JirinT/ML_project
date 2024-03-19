import json
import os
import time

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import folder_functions

from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score

from preprocessing.simple_preprocessor import SimplePreprocessor
from datasets.simple_dataloader import SimpleDataLoader


def decode_labels(labels):
	
	flow_rate = np.zeros((3,))
	lateral_speed = np.zeros((3,))
	z_offset = np.zeros((3,))
	hotend_temperature = np.zeros((3,))

	for i in range(labels.shape[0]):
		flow_rate = np.vstack((flow_rate,labels[i][:3]))
		lateral_speed = np.vstack((lateral_speed,labels[i][3:6]))
		z_offset = np.vstack((z_offset,labels[i][6:9]))
		hotend_temperature = np.vstack((hotend_temperature,labels[i][9:]))

	flow_rate_decoded = np.argmax(flow_rate, axis=1) + 1
	lateral_speed_decoded = np.argmax(lateral_speed, axis=1) + 1
	z_offset_decoded = np.argmax(z_offset, axis=1) + 1
	hotend_temperature_decoded = np.argmax(hotend_temperature, axis=1) + 1

	return flow_rate_decoded, lateral_speed_decoded, z_offset_decoded, hotend_temperature_decoded


def apply_cross_validation(trainX, trainY, config):

	print("Cross validation started.")
	k_range = range(1,config["training"]["num_k"])
	k_accuracy = []

	with open(os.path.join(log_folder_training, "log.txt"), "w") as file:
		for k in k_range:
			print(f"Now running for k = {k}")
			knn = KNeighborsClassifier(n_neighbors=k, metric=config["classifier"]["distance_metric"])

			accuracies = cross_val_score(knn, X=trainX, y=trainY, cv=config["training"]["cv_fold"])
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

	return best_k


def apply_grid_search_cv(knn, trainX, trainY, config):
	
	print("Starting grid search...")

	param_grid = {
		'n_neighbors': range(1, config["training"]["num_k"]+1),
		'metric': ['euclidean', 'manhattan', 'chebyshev']
	}

	grid_search = GridSearchCV(knn, param_grid, cv=config["training"]["cv_fold"])

	grid_search.fit(trainX, trainY)

	print("Best Parameters:", grid_search.best_params_)
	print("Best Score:", grid_search.best_score_)

	return grid_search


def apply_own_grid_search(trainX, trainY, valX, valY, config):

	print("Starting grid search...")

	param_grid = {
		'n_neighbors': range(1, config["training"]["num_k"]+1),
		'metric': ['euclidean', 'manhattan', 'chebyshev'],
		"weights": ["uniform", "distance"]
	}

	best_accuracy = 0
	best_params = {'n_neighbors': 1, 'metric': "euclidean", 'weight': "uniform"}

	accuracy_table = {'Metric': [], 'k': [], 'Weight': [], 'Accuracy': []}

	for k in param_grid["n_neighbors"]:
		for metric in param_grid["metric"]:
			for weight in param_grid["weights"]:
				print(f"Evaluating k={k}, metric={metric}, weight={weight}...")
				knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
				knn.fit(trainX, trainY)
				
				validation_accuracy = knn.score(valX, valY)

				accuracy_table['Metric'].append(metric)
				accuracy_table['k'].append(k)
				accuracy_table['Weight'].append(weight)
				accuracy_table['Accuracy'].append(validation_accuracy)
				
				if validation_accuracy > best_accuracy:
					best_accuracy = validation_accuracy
					best_params = {'n_neighbors': k, 'metric': metric, 'weight': weight}

	accuracy_df = pd.DataFrame(accuracy_table)

	accuracy_df.to_csv(os.path.join(log_folder_training, "grid_search.csv"), sep='\t', index=False)

	print("Best Parameters:", best_params)
	print("Best Score:", best_accuracy)

	return best_params


def apply_PCA(trainX, valX, testX, config):

	pca = PCA(n_components=config["training"]["pca_components"])
	trainX = pca.fit_transform(trainX)
	valX = pca.transform(valX)
	testX = pca.transform(testX)

	return trainX, valX, testX


def log_classified_samples(testX, testY, y_predicted, config):

	correct_classification = np.zeros(shape=(testX.shape[1],))
	incorrect_classification = np.zeros(shape=(testX.shape[1],))

	for i in range(testY.shape[0]):
		comparison = np.array_equal(testY[i], y_predicted[i])
		if comparison == True:
			correct_classification = np.hstack((correct_classification,testX[i]))
		else:
			incorrect_classification = np.hstack((incorrect_classification,testX[i]))

	correct_classification = correct_classification.reshape(-1, config["preprocessor"]["resize"]["height"], config["preprocessor"]["resize"]["width"])
	incorrect_classification = incorrect_classification.reshape(-1, config["preprocessor"]["resize"]["height"], config["preprocessor"]["resize"]["width"])

	correct_class_path = config["general"]["classified_images_path"]["correct"]
	incorrect_class_path = config["general"]["classified_images_path"]["incorrect"]

	folder_functions.create_folder(correct_class_path)
	folder_functions.create_folder(incorrect_class_path)

	folder_functions.delete_files(correct_class_path)
	folder_functions.delete_files(incorrect_class_path)

	folder_functions.save_images(correct_class_path, correct_classification)
	folder_functions.save_images(incorrect_class_path, incorrect_classification)


def log_confusion_matrix(testY, y_predicted, config):

	flow_rate_test_decoded, lateral_speed_test_decoded, z_offset_test_decoded, hotend_temperature_test_decoded = decode_labels(testY)
	flow_rate_predicted_decoded, lateral_speed_predicted_decoded, z_offset_predicted_decoded, hotend_temperature_predicted_decoded = decode_labels(y_predicted)

	# Plot confusion matrix
	fig, axs = plt.subplots(2,2,figsize=(8, 5))
	cmp1 = ConfusionMatrixDisplay(confusion_matrix(flow_rate_test_decoded, flow_rate_predicted_decoded),
								display_labels=["Low", "Good", "High"])
	cmp1.plot(ax=axs[0, 0])
	axs[0, 0].set_title('Flow Rate')

	cmp2 = ConfusionMatrixDisplay(confusion_matrix(lateral_speed_test_decoded, lateral_speed_predicted_decoded),
								display_labels=["Low", "Good", "High"])
	cmp2.plot(ax=axs[0, 1])
	axs[0, 1].set_title('Lateral Speed')

	cmp3 = ConfusionMatrixDisplay(confusion_matrix(z_offset_test_decoded, z_offset_predicted_decoded),
								display_labels=["Low", "Good", "High"])
	cmp3.plot(ax=axs[1, 0])
	axs[1, 0].set_title('Z offset')

	cmp4 = ConfusionMatrixDisplay(confusion_matrix(hotend_temperature_test_decoded, hotend_temperature_predicted_decoded),
								display_labels=["Low", "Good", "High"])
	cmp4.plot(ax=axs[1, 1])
	axs[1, 1].set_title('Hotend Temperature')
	plt.tight_layout()

	conf_matrix_path = config["general"]["conf_matrix_path"]
	folder_functions.create_folder(conf_matrix_path)

	timestamp = datetime.now()
	timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
	filename = f'confusion_matrices_{timestamp}.png'

	plt.savefig(os.path.join(conf_matrix_path, filename))

	flow_rate_acc = accuracy_score(flow_rate_test_decoded, flow_rate_predicted_decoded)
	lateral_speed_acc = accuracy_score(lateral_speed_test_decoded, lateral_speed_predicted_decoded)
	z_offset_acc = accuracy_score(z_offset_test_decoded, z_offset_predicted_decoded)
	hotend_temperature_acc = accuracy_score(hotend_temperature_test_decoded, hotend_temperature_predicted_decoded)

	with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
		file.write(f"\nFlow rate accuracy: {round(flow_rate_acc * 100, 2)} %\n"
				f"Lateral speed accuracy: {round(lateral_speed_acc * 100, 2)} %\n"
				f"Z offset accuracy: {round(z_offset_acc * 100, 2)} %\n"
				f"Hot end temperature accuracy: {round(hotend_temperature_acc * 100, 2)} %\n")

	print("Accuracy saved to log files.")


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

user = config["active_user"]
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
folder_functions.create_folder(config["general"]["sample_img_path"])
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

(trainX, valX, trainY, valY) = train_test_split(
	trainX, 
	trainY, 
	test_size=config["training"]["val_size"], 
	random_state=config["training"]["random_state"]
	)

if config["training"]["use_normalization"]:
	print("Applying MinMaxScaler...")
	scaler = MinMaxScaler()
	trainX = scaler.fit_transform(trainX)
	testX = scaler.transform(testX)

if config["training"]["use_pca"]:
	print("Applying PCA...")
	trainX, valX, testX = apply_PCA(trainX, valX, testX, config)

if config["training"]["use_cross_validation"]:
	best_k = apply_cross_validation(trainX, trainY, config)
	knn = KNeighborsClassifier(n_neighbors=best_k, metric=config["classifier"]["distance_metric"])
	knn.fit(trainX, trainY)
elif config["training"]["use_grid_search"]:
	best_params = apply_own_grid_search(trainX, trainY, valX, valY, config)
	knn = KNeighborsClassifier(n_neighbors=best_params["n_neighbors"], metric=best_params["metric"], weights=best_params["weight"])
	knn.fit(trainX, trainY)
else:
	k = config["classifier"]["k_value"]
	knn = KNeighborsClassifier(n_neighbors=k, metric=config["classifier"]["distance_metric"])
	knn.fit(trainX, trainY)

y_predicted = knn.predict(testX)
test_accuracy = knn.score(testX, testY)

print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
	file.write("\nTest Accuracy: {:.2f}%".format(test_accuracy * 100))

if config["general"]["log_classified_images"]:
	log_classified_samples(testX, testY, y_predicted, config)

if config["general"]["log_confusion_matrix"]:
	log_confusion_matrix(testY, y_predicted, config)