import pandas as pd
import numpy as np

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

train_data = pd.read_csv('census-income.data.csv', header=None, delimiter=' ', names=column_names)

# Process first 14 columns
columns_to_process = train_data.columns[:14]
# Remove trailing commas in the first 14 columns
train_data[columns_to_process] = train_data[columns_to_process].apply(lambda x: x.str.replace(',', ''))
# Turn ? into NaN
train_data.replace('?', np.nan, inplace=True)
# Turn <=50K: 0, >50K: 1
train_data['income'] = train_data['income'].replace({'<=50K': 0, '>50K': 1})
# Remove education-num column because it is redundant (it is the numerical version of education)
train_data.drop('education', axis=1, inplace=True)
# Remove rows with missing values
train_data.dropna(inplace=True)

imputed_data = train_data.copy()

from sklearn.preprocessing import LabelEncoder
categorical_columns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Iterate through each categorical column and apply label encoding
for column in categorical_columns:
    imputed_data[column] = label_encoder.fit_transform(imputed_data[column])

# Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

X = imputed_data.drop('income', axis=1)  # Features
y = imputed_data['income']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
