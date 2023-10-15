import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.stats import mode

class K_Nearest_Neighbors_Classifier() :
    def __init__( self, K ) :	
        self.K = K
        
    # Function to store training set
    def fit( self, X_train, Y_train ) :		
        self.X_train = X_train		
        self.Y_train = Y_train						
        self.m, self.n = X_train.shape # no_of_training_examples, no_of_features
            
    # Function for prediction		
    def myKNN( self, X_test ) :		
        self.X_test = X_test						
        self.m_test, self.n = X_test.shape # no_of_test_examples, no_of_features	 
        # initialize Y_predict		
        Y_predict = np.zeros( self.m_test )	
        for i in range( self.m_test ) :			
            x = self.X_test[i]			
            # find the K nearest neighbors from current test example			
            neighbors = np.zeros( self.K )			
            neighbors = self.find_neighbors( x )			
            # most frequent class in K neighbors			
            Y_predict[i] = mode( neighbors )[0][0]				
        return Y_predict
    
    # Function to find the K nearest neighbors to current test example			
    def find_neighbors( self, x ) :		
        # calculate all the euclidean distances between current
        # test example x and training set X_train		
        euclidean_distances = np.zeros( self.m )		
        for i in range( self.m ) :			
            d = self.euclidean( x, self.X_train[i] )			
            euclidean_distances[i] = d		
        # sort Y_train according to euclidean_distance_array and
        # store into Y_train_sorted		
        inds = euclidean_distances.argsort()	
        Y_train_sorted = self.Y_train[inds]		
        return Y_train_sorted[:self.K]
    
    # Function to calculate euclidean distance			
    def euclidean( self, x, x_train ) :		
        return np.sqrt( np.sum( np.square( x - x_train ) ) )

def main() :	
    # Importing dataset	and shuffle the rows
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, 'CTG.csv')    
    df = pd.read_csv(file_path)
    df = df.drop("CLASS", axis=1)
    shuffled_df = df.sample(frac=1, random_state=0)
    X = shuffled_df.iloc[:,:-1].values
    Y = shuffled_df.iloc[:,-1:].values
    
    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 1/3, random_state = 0 )

    # Feature normalization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    # Model training
    model_k5 = K_Nearest_Neighbors_Classifier( K = 5 )	
    model_k5.fit( X_train, Y_train )	
    model_k10 = K_Nearest_Neighbors_Classifier( K = 10 )	
    model_k10.fit( X_train, Y_train )
    model_k50 = K_Nearest_Neighbors_Classifier( K = 50 )	
    model_k50.fit( X_train, Y_train )
    
    # Prediction on test set
    Y_pred_k5 = model_k5.myKNN( X_test )
    Y_pred_k10 = model_k10.myKNN( X_test )
    Y_pred_k50 = model_k50.myKNN( X_test )
    
    # measure performance	
    correctly_classified_k5 = 0	
    correctly_classified_k10 = 0
    correctly_classified_k50 = 0
    
    # counter	
    count = 0	
    for count in range( np.size( Y_pred_k5 ) ) :		
        if Y_test[count] == Y_pred_k5[count] :			
            correctly_classified_k5 += 1		
        if Y_test[count] == Y_pred_k10[count] :			
            correctly_classified_k10 += 1
        if Y_test[count] == Y_pred_k50[count] :			
            correctly_classified_k50 += 1			
        count = count + 1
        
    print( "Accuracy for k = 5  : ", (
    correctly_classified_k5 / count ) * 100 )
    print( "Accuracy for k = 10 : ", (
    correctly_classified_k10 / count ) * 100 )
    print( "Accuracy for k = 50 : ", (
    correctly_classified_k50 / count ) * 100 )
    
    # Creating the confusion matrix
    Y_pred_k5 = Y_pred_k5.reshape(Y_test.shape)
    unique_classes = np.unique(np.concatenate((Y_test, Y_pred_k5)))
    num_classes = len(unique_classes)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(Y_test)):
        true_class_idx = np.where(unique_classes == Y_test[i])[0][0]
        pred_class_idx = np.where(unique_classes == Y_pred_k5[i])[0][0]
        confusion_matrix[true_class_idx][pred_class_idx] += 1
    
    # Printing the confusion matrix
    print("Confusion Matrix:")

    print("\t" + "\t".join(map(str, unique_classes.astype(int))))
    for i in range(num_classes):
        print(unique_classes[i].astype(int), end="\t")
        for j in range(num_classes):
            print(confusion_matrix[i][j], end="\t")
        print()
    print("(Rows are actual classes, columns are predicted classes)")

if __name__ == "__main__" :
    main()
