Implement the K-Nearest Neighbors (KNN) classifier algorithm (from scratch) and apply to the Cardiotocography (CTG) dataset to pedict the class of test data based on the closest k training examples in the feature space.

Features:
- Custom KNN Implementation: The project contains a K_Nearest_Neighbors_Classifier class with methods to fit the model on training data and predict on test data.
- Data Preprocessing: Before feeding the data to the model, the dataset is imported, shuffled, and split into training and test sets. Features are normalized using the mean and standard deviation of the training data.
- Model Training: Multiple KNN models are trained with different values of k (i.e., 5, 10, and 50).
- Performance Evaluation: The performance of each model is evaluated on the test set by calculating the accuracy.
- Confusion Matrix: A confusion matrix is generated to provide insights into the true positive, true negative, false positive, and false negative predictions.
