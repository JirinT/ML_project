import json
import os
import time

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import folder_functions

from datetime import datetime
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

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
	k_range = range(1,config["knn"]["training"]["num_k"])
	k_accuracy = []

	with open(os.path.join(log_folder_training, "log.txt"), "w") as file:
		for k in k_range:
			print(f"Now running for k = {k}")
			knn = KNeighborsClassifier(n_neighbors=k, metric=config["knn"]["classifier"]["distance_metric"])

			accuracies = cross_val_score(knn, X=trainX, y=trainY, cv=config["knn"]["training"]["cv_fold"])
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


def evaluate_knn(params, trainX, trainY, valX, valY):

	k = params['k']
	metric = params['metric']
	weight = params['weight']
	print(f"Evaluating k={k}, metric={metric}, weight={weight}...")
	knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
	knn.fit(trainX, trainY)

	validation_accuracy = knn.score(valX, valY)
	f1_each_label = f1_score(valY, knn.predict(valX), average=None)
	f1 = np.mean(f1_each_label)

	return {'k': k, 'metric': metric, 'weight': weight, 'accuracy': validation_accuracy, 'f1_score_avg': f1, 'f1_score_each': f1_each_label}


def apply_own_grid_search(trainX, trainY, valX, valY, config, label):

	print(f"Starting grid search for label: {label} ...")

	param_grid = {
		'n_neighbors': range(1, config["knn"]["training"]["num_k"]+1),
		'metric': ['euclidean', 'manhattan', 'chebyshev'],
		"weights": ["uniform", "distance"]
	}

	param_combinations = [{'k': k, 'metric': metric, 'weight': weight} for k in param_grid["n_neighbors"]
                      for metric in param_grid["metric"]
                      for weight in param_grid["weights"]]

	results = Parallel(n_jobs=-1)(delayed(evaluate_knn)(params, trainX, trainY, valX, valY) for params in param_combinations)

	accuracy_df = pd.DataFrame(results)

	best_accuracy = max(result['accuracy'] for result in results)
	best_f1_score = max(result['f1_score_avg'] for result in results)

	if config["knn"]["training"]["optimizer_metric"]["accuracy"]:
		best_params = [result for result in results if result['accuracy'] == best_accuracy][0]
	else:
		best_params = [result for result in results if result['f1_score_avg'] == best_f1_score][0]

	accuracy_df.to_csv(os.path.join(log_folder_training, f"grid_search_{label}.csv"), sep='\t', index=False)
	with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
		file.write(f"Grid search for {label}:\n")
		file.write("\tBest Parameters: " + str(best_params) + "\n")
		file.write("\tBest Validation set Accuracy: {:.2f}%".format(best_accuracy * 100) + "\n")
		file.write("\tBest Validation set average F1 Score: {:.2f}%".format(best_f1_score * 100) + "\n")
		file.write("\n\tF1 Score for each label:\n")
		file.write("\t\tLow: {:.2f}%\n".format(best_params["f1_score_each"][0] * 100))
		file.write("\t\tGood: {:.2f}%\n".format(best_params["f1_score_each"][1] * 100))
		file.write("\t\tHigh: {:.2f}%\n\n".format(best_params["f1_score_each"][2] * 100))
	return best_params


def apply_PCA(trainX, valX, testX, config):

	pca = PCA(n_components=config["knn"]["training"]["pca_components"])
	print("number of components:", pca.n_components_)
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
	_, axs = plt.subplots(2,2,figsize=(8, 5))
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

def histograms(labels, config):
	# HISTOGRAM for every combination of labels
	# Convert each vector to a tuple and then to a string
	label_strings = [''.join(map(str, v.ravel())) for v in labels]
	# Get unique labels and their counts
	unique_labels, counts = np.unique(label_strings, return_counts=True)

	# Create the histogram
	plt.figure("Histogram of Unique Labels", figsize=(10, 5))
	plt.bar(range(len(unique_labels)), counts)
	plt.xlabel("Unique Labels")
	plt.ylabel("Amount of samples")
	plt.title('Histogram of Unique Labels')

	# save the histogram with timestamp:
	folder_functions.create_folder(config["general"]["histogram_path"])
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	plt.savefig(os.path.join(config["general"]["histogram_path"], f"histogram_all_labels_{timestamp}.png"))

	# HISTOGRAM for every label
	flow_rate_labels = labels[:, 0:3]
	lateral_speed_labels = labels[:, 3:6]
	z_offset_labels = labels[:, 6:9]
	hotend_temperature_labels = labels[:, 9:]

	unique_flow_rate, counts_flow_rate = np.unique(flow_rate_labels, return_counts=True, axis=0)
	unique_lateral_speed, counts_lateral_speed = np.unique(lateral_speed_labels, return_counts=True, axis=0)
	unique_z_offset, counts_z_offset = np.unique(z_offset_labels, return_counts=True, axis=0)
	unique_hotend_temperature, counts_hotend_temperature = np.unique(hotend_temperature_labels, return_counts=True, axis=0)

	plt.figure("Histograms for each label", figsize=(13, 8))
	plt.subplot(2, 2, 1)
	plt.bar(range(len(unique_flow_rate)), counts_flow_rate)
	plt.xticks([0, 1, 2], ['High', 'Good', 'Low'])
	plt.ylabel("Amount of samples")
	plt.title("Flow Rate")

	plt.subplot(2, 2, 2)
	plt.bar(range(len(unique_lateral_speed)), counts_lateral_speed)
	plt.xticks([0, 1, 2], ['High', 'Good', 'Low'])
	plt.ylabel("Amount of samples")
	plt.title("Lateral Speed")

	plt.subplot(2, 2, 3)
	plt.bar(range(len(unique_z_offset)), counts_z_offset)
	plt.xticks([0, 1, 2], ['High', 'Good', 'Low'])
	plt.ylabel("Amount of samples")
	plt.title("Z Offset")

	plt.subplot(2, 2, 4)
	plt.bar(range(len(unique_hotend_temperature)), counts_hotend_temperature)
	plt.xticks([0, 1, 2], ['High', 'Good', 'Low'])
	plt.ylabel("Amount of samples")
	plt.title("Hotend Temperature")

	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	plt.savefig(os.path.join(config["general"]["histogram_path"], f"histogram_separate_{timestamp}.png"))


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

# Define the active user and his data path
user = config["active_user"]
data_path = config["general"]["data_paths"][user]

# Initialize the preprocessor and dataloader
simple_preprocessor = SimplePreprocessor(
	width=config["preprocessor"]["resize"]["width"], 
	height=config["preprocessor"]["resize"]["height"]
	)
dataloader = SimpleDataLoader(data_path, preprocessors=simple_preprocessor)

# Load the data
data, labels = dataloader.load_data(
	num_samples_subset=config["knn"]["training"]["num_samples_subset"], 
	start_idx=config["knn"]["training"]["start_idx"], 
	end_idx=config["knn"]["training"]["end_idx"]
	)
if config["general"]["log_histograms"] and user == "remote_pc":
	histograms(labels, config) # create histograms for the labels

# Flatten the data and labels
imgs_flat = data.reshape(data.shape[0], -1) # flatten the image matrix to 1D vector
labels_flat = labels.reshape(labels.shape[0], -1) # flatten the labels matrix to 1D vector

# Save sample images
show_images = config["general"]["show_sample_images"]
folder_functions.create_folder(config["general"]["sample_img_path"])
if show_images and user == "remote_pc":
	for img_idx in range(len(data)):
		if img_idx > 5:
			break
		cv.imwrite(os.path.join(config["general"]["sample_img_path"], f"sample_image_{img_idx}.png"), data[img_idx])

# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(
	imgs_flat, 
	labels_flat, 
	test_size=config["knn"]["training"]["test_size"], 
	random_state=config["knn"]["training"]["random_state"]
	)

# Split the training set into training and validation sets
(trainX, valX, trainY, valY) = train_test_split(
	trainX, 
	trainY, 
	test_size=config["knn"]["training"]["val_size"], 
	random_state=config["knn"]["training"]["random_state"]
	)

# Apply normalization
if config["knn"]["training"]["use_normalization"]:
	print("Applying MinMaxScaler...")
	scaler = MinMaxScaler()
	trainX = scaler.fit_transform(trainX)
	testX = scaler.transform(testX)

# Apply PCA
if config["knn"]["training"]["use_pca"]:
	print("Applying PCA...")
	trainX, valX, testX = apply_PCA(trainX, valX, testX, config)

# Find best setting for KNN using cross-validation or grid search
if config["knn"]["training"]["use_cross_validation"]:
	best_k = apply_cross_validation(trainX, trainY, config)
	knn = KNeighborsClassifier(n_neighbors=best_k, metric=config["knn"]["classifier"]["distance_metric"])
	knn.fit(trainX, trainY)
elif config["knn"]["training"]["use_grid_search"]:
	start_time = time.time()

	if config["knn"]["training"]["knn_all_in_one"]:
		best_params_all = apply_own_grid_search(trainX, trainY, valX, valY, config, label="all")
		knn_all = KNeighborsClassifier(n_neighbors=best_params_all["k"], metric=best_params_all["metric"], weights=best_params_all["weight"])
		knn_all.fit(trainX, trainY)
	else:
		best_params_flow_rate = apply_own_grid_search(trainX, trainY[:, 0:3], valX, valY[:, 0:3], config, label="flow_rate")
		best_params_lateral_speed = apply_own_grid_search(trainX, trainY[:, 3:6], valX, valY[:, 3:6], config, label="lateral_speed")
		best_params_z_offset = apply_own_grid_search(trainX, trainY[:, 6:9], valX, valY[:, 6:9], config, label="z_offset")
		best_params_hotend_temperature = apply_own_grid_search(trainX, trainY[:, 9:], valX, valY[:, 9:], config, label="hotend_temperature")
	
		knn_flow_rate = KNeighborsClassifier(n_neighbors=best_params_flow_rate["k"], metric=best_params_flow_rate["metric"], weights=best_params_flow_rate["weight"])
		knn_lateral_speed = KNeighborsClassifier(n_neighbors=best_params_lateral_speed["k"], metric=best_params_lateral_speed["metric"], weights=best_params_lateral_speed["weight"])
		knn_z_offset = KNeighborsClassifier(n_neighbors=best_params_z_offset["k"], metric=best_params_z_offset["metric"], weights=best_params_z_offset["weight"])
		knn_hotend_temperature = KNeighborsClassifier(n_neighbors=best_params_hotend_temperature["k"], metric=best_params_hotend_temperature["metric"], weights=best_params_hotend_temperature["weight"])

		knn_flow_rate.fit(trainX, trainY[:,0:3])
		knn_lateral_speed.fit(trainX, trainY[:,3:6])
		knn_z_offset.fit(trainX, trainY[:,6:9])
		knn_hotend_temperature.fit(trainX, trainY[:,9:])

	with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
		file.write("\nGrid search took {:.2f} seconds.\n".format(time.time() - start_time))
else:
	k = config["knn"]["classifier"]["k_value"]
	if config["knn"]["training"]["knn_all_in_one"]:
		knn_all = KNeighborsClassifier(n_neighbors=k, metric=config["knn"]["classifier"]["distance_metric"])
		knn_all.fit(trainX, trainY)
	else:
		knn_flow_rate = KNeighborsClassifier(n_neighbors=k, metric=config["knn"]["classifier"]["distance_metric"])
		knn_lateral_speed = KNeighborsClassifier(n_neighbors=k, metric=config["knn"]["classifier"]["distance_metric"])
		knn_z_offset = KNeighborsClassifier(n_neighbors=k, metric=config["knn"]["classifier"]["distance_metric"])
		knn_hotend_temperature = KNeighborsClassifier(n_neighbors=k, metric=config["knn"]["classifier"]["distance_metric"])

		knn_flow_rate.fit(trainX, trainY[:, 0:3])
		knn_lateral_speed.fit(trainX, trainY[:, 3:6])
		knn_z_offset.fit(trainX, trainY[:, 6:9])
		knn_hotend_temperature.fit(trainX, trainY[:, 9:])

if config["knn"]["training"]["knn_all_in_one"]:
	y_predicted = knn_all.predict(testX)
	test_accuracy_all = knn_all.score(testX, testY)
	f1_score_all = f1_score(testY, y_predicted, average=None)
	f1_score_all = np.mean(f1_score_all)
	with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
		file.write("\nTest Accuracy All Classes: {:.2f}%".format(test_accuracy_all * 100))
		file.write("\nF1 Score All Classes: {:.2f}%".format(f1_score_all * 100))
else:
	y_predicted_flow_rate = knn_flow_rate.predict(testX)
	y_predicted_lateral_speed = knn_lateral_speed.predict(testX)
	y_predicted_z_offset = knn_z_offset.predict(testX)
	y_predicted_hotend_temperature = knn_hotend_temperature.predict(testX)

	test_accuracy_flow_rate = knn_flow_rate.score(testX, testY[:, 0:3])
	test_accuracy_lateral_speed = knn_lateral_speed.score(testX, testY[:, 3:6])
	test_accuracy_z_offset = knn_z_offset.score(testX, testY[:, 6:9])
	test_accuracy_hotend_temperature = knn_hotend_temperature.score(testX, testY[:, 9:])

	f1_score_flow_rate = f1_score(testY[:, 0:3], y_predicted_flow_rate, average=None)
	f1_score_flow_rate = np.mean(f1_score_flow_rate)
	f1_score_lateral_speed = f1_score(testY[:, 3:6], y_predicted_lateral_speed, average=None)
	f1_score_lateral_speed = np.mean(f1_score_lateral_speed)
	f1_score_z_offset = f1_score(testY[:, 6:9], y_predicted_z_offset, average=None)
	f1_score_z_offset = np.mean(f1_score_z_offset)
	f1_score_hotend_temperature = f1_score(testY[:, 9:], y_predicted_hotend_temperature, average=None)
	f1_score_hotend_temperature = np.mean(f1_score_hotend_temperature)

	with open(os.path.join(log_folder_training, "log.txt"), "a") as file:
		file.write("\nTest accuracies:\n")
		file.write("\tFlow Rate: {:.2f}%\n".format(test_accuracy_flow_rate * 100))
		file.write("\tLateral Speed: {:.2f}%\n".format(test_accuracy_lateral_speed * 100))
		file.write("\tZ Offset: {:.2f}%\n".format(test_accuracy_z_offset * 100))
		file.write("\tHotend Temperature: {:.2f}%\n".format(test_accuracy_hotend_temperature * 100))
		file.write("\nTest F1 scores (average values):\n")
		file.write("\tFlow Rate: {:.2f}%\n".format(f1_score_flow_rate * 100))
		file.write("\tLateral Speed: {:.2f}%\n".format(f1_score_lateral_speed * 100))
		file.write("\tZ Offset: {:.2f}%\n".format(f1_score_z_offset * 100))
		file.write("\tHotend Temperature: {:.2f}%\n".format(f1_score_hotend_temperature * 100))

if not config["knn"]["training"]["knn_all_in_one"]:
	y_predicted = np.hstack((y_predicted_flow_rate, y_predicted_lateral_speed, y_predicted_z_offset, y_predicted_hotend_temperature))

if config["general"]["log_classified_images"] and user == "remote_pc":
	log_classified_samples(testX, testY, y_predicted, config)

if config["general"]["log_confusion_matrix"] and user == "remote_pc":
	log_confusion_matrix(testY, y_predicted, config)

print("End of KNN classification. Logs saved to log folder.")