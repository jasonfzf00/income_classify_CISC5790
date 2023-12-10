# Data processing
import pandas as pd
import os
import preprocess as pre

# Modelling
from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Visualization and model evaluation
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

def random_forest(df):
    x = df.drop(columns=['fnlwgt', 'income'], axis=1)
    y = df['income']
    
    ros = RandomOverSampler()
    ros.fit(x, y)
    X_resampled, Y_resampled = ros.fit_resample(x, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2)
    rf = RandomForestClassifier()
    param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

    # Create a random forest classifier
    rf = RandomForestClassifier()
    
    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=5, 
                                    cv=5)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)
        
    # Create a variable for the best model
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  rand_search.best_params_)
    
    # Generate predictions with the best model
    y_pred = best_rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()
    
    # Print the assessment
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

def main():
    #Read the data
    df = pd.read_csv(os.getcwd()+'/data/census-income-data-cleaned.csv')
    random_forest(df)
    
    
if __name__ == "__main__":
    main()