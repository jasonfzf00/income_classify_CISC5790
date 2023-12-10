import pandas as pd
import numpy as np

# Load cleaned training data
train_data = pd.read_csv('census-income.data.csv')

# Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

X_train_data = train_data.drop('income', axis=1)  # Features
y_train_data = train_data['income']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size=0.2, random_state=42)

# Initialize the Naive Bayes model
naive_bayes_model = GaussianNB()
# Train the model on the training set
naive_bayes_model.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = naive_bayes_model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report_output)

# Test file
# Load cleaned testing data
test_data = pd.read_csv('census-income.test.csv')

# Drop target variable
X_test_data = test_data.drop('income', axis=1)
y_test_data = test_data['income']

# Make predictions using the trained Naive Bayes model
y_pred_test = naive_bayes_model.predict(X_test_data)

# Evaluate the model on the test data
accuracy_test = accuracy_score(y_test_data, y_pred_test)
classification_report_test = classification_report(y_test_data, y_pred_test)

print(f"Test Data Accuracy: {accuracy_test}")
print("Test Data Classification Report:")
print(classification_report_test)


