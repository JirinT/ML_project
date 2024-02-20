# Machine learning project
Repository for the ml project.

<ul>
<li>The .csv file with the dataset can not be uploaded to github, because it is too large. So we have to have the .csv in local directory.</li>
</ul>

Project stages:
	1. Find a topic/project:
		a. Decide for classification or regression
	2. Reduce number of samples in dataset
		a. Random approach
		b. Prob around 100k samples
			i. Do train, val, test split with this set
	3. Dataloading/Image Preprocessing
		a. Crop around nozzle (240x240)
		b. Resize the image without quality loss
			i. Look up some strategies
		c. Transform image to grayscale
	-> this happens during dataloading
	4. Build the KNN model
		a. Select appropiate k
		b. Select appropiate distance metric
	5. Evaluate KNN model
		a. Evaluation metrics
		b. Visualization graphs
		c. Table for different k
	6. Optimization
		a. Find optimal k
			i. Cross-validation
	7. Build the CNN model
	8. Evaluate the CNN
	9. Optimize the CNN!
