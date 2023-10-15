### KNN
**Implement the K-Nearest Neighbors (KNN) classifier algorithm (from scratch)** and apply to the Cardiotocography (CTG) dataset to pedict the class of test data based on the closest k training examples in the feature space.

Features:
- Custom KNN Implementation: K_Nearest_Neighbors_Classifier class with methods to fit the model on training data and predict on test data.
- Data Preprocessing: Before feeding the data to the model, the dataset is imported, shuffled, and split into training and test sets. Features are normalized using the mean and standard deviation of the training data.
- Model Training: Multiple KNN models are trained with different values of k (i.e., 5, 10, and 50).
- Performance Evaluation: Evaluate on the test set by calculating the accuracy.
- Confusion Matrix: To provide insights into the true positive, true negative, false positive, and false negative predictions.

Output:
Accuracy for k = 5  :  89.28067700987306
Accuracy for k = 10 :  88.15232722143864
Accuracy for k = 50 :  87.58815232722144
Confusion Matrix:
        1       2       3
1       533     17      1
2       43      54      2
3       4       9       46
