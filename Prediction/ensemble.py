# File Processing
import pandas as pd
import numpy as np
import os as os

# Evaluation
from sklearn.metrics import accuracy_score,precision_score, recall_score

# Set path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read data files
test_data = pd.read_csv(os.path.dirname(os.getcwd())+'/Handling Missing Values/census-income.test.csv')
test_y = test_data['income']
knn_predict = pd.read_csv('KNNPredictionTest.csv')
logistic_predict = pd.read_csv('LogisticPredictionTest.csv')
rf_predict = pd.read_csv('RandomForestPredictionTest.csv')
nb_predict = pd.read_csv('NBPredictionTest.csv')

# Find accuracy for each models:
print('Accuracy for KNN: ', accuracy_score(test_y, knn_predict))
print('Accuracy for Logistic Regression: ', accuracy_score(test_y, logistic_predict))
print('Accuracy for Random Forest: ', accuracy_score(test_y, rf_predict))
print('Accuracy for Naive Bayes: ', accuracy_score(test_y, nb_predict),'\n')

# Combine result
combined_predict = pd.concat([knn_predict, logistic_predict,rf_predict,nb_predict], axis=1)

# Find the majority vote
majority_vote = combined_predict.mode(axis=1, dropna=True).iloc[:, 0]

# Accuracy for majority vote
print('Accuracy after Majority Vote: ', accuracy_score(test_y, majority_vote))