# Data processing
import pandas as pd
import os

# Modeling
from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
import joblib

# Visualization and model evaluation
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score, ConfusionMatrixDisplay

def simple_rf(df):
    x = df.drop(columns=['fnlwgt', 'income'], axis=1)
    y = df['income']
    
    # Fix the problem with imbalanced training dataset
    ros = RandomOverSampler()
    ros.fit(x, y)
    X_resampled, Y_resampled = ros.fit_resample(x, y)
        
    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Fit the random search object to the data
    rf.fit(X_resampled, Y_resampled)
    
    joblib.dump(rf, 's_rf.joblib')
    
    return rf

def build_random_forest(df):
    x = df.drop(columns=['fnlwgt', 'income'], axis=1)
    y = df['income']
    
    # Fix the problem with imbalanced training dataset
    ros = RandomOverSampler()
    ros.fit(x, y)
    X_resampled, Y_resampled = ros.fit_resample(x, y)
    
    #X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2) # Folding if no test files
    
    # Create a random forest classifier
    rf = RandomForestClassifier()
    
    # Use random search to find the best hyperparameters, an ensemble method
    param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=10, 
                                    cv=5)

    # Fit the random search object to the data
    rand_search.fit(X_resampled, Y_resampled)
        
    # Create a variable for the best model
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  rand_search.best_params_)
    
    joblib.dump(best_rf, 'rf.joblib')
    
    return best_rf

# Make predictions and evaluate model. Set write_result to True if you want to write result.
def model_evaluation(test_df, rf, write_result = False):
    x = test_df.drop(columns=['fnlwgt', 'income'], axis=1)
    y = test_df['income']
    
    #Use provided Random Forest Classifier to predict for given y
    y_pred = rf.predict(x)
    
    if write_result == True:
        output = pd.DataFrame(y_pred)
        output.to_csv(os.path.dirname(os.getcwd()) + '/Prediction/RandomForestPredictionTest.csv',header=['RandomForest_predictions'],index=False)
        

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y,y_pred)

    # Create the confusion matrix
    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()
    
    # Print the assessment
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

def main():
    # Read the data
    file_path = os.path.dirname(os.getcwd())
    
    df_train = pd.read_csv(file_path+'/Handling Missing Values/census-income.data.csv')
    df_test = pd.read_csv(file_path+'/Handling Missing Values/census-income.test.csv')
    
    if not os.path.exists(file_path+'/randomForest/rf.joblib'):
        s_rf = simple_rf(df_train)
    else:
        s_rf = joblib.load('s_rf.joblib')
    
    #s_rf = simple_rf(df_train)
    
    # Evaluate Train dataset
    model_evaluation(df_train,s_rf)
    
    # Evaluate Test dataset
    model_evaluation(df_test,s_rf)

    # Check if model exists, build if not
    if not os.path.exists(file_path+'/randomForest/rf.joblib'):
        rf = build_random_forest(df_train)
    else:
        rf = joblib.load('rf.joblib')
    
    #print(rf.feature_importances_)
    
    # Evaluate Train dataset
    model_evaluation(df_train,rf)
    
    # Evaluate Test dataset
    model_evaluation(df_test,rf,True)
    
if __name__ == "__main__":
    main()