import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

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
# Drop education and fnlwgt
train_data = train_data.drop(['education', 'fnlwgt'], axis=1)

# Impute missing values using forward fill
imputed_data = train_data.copy()
imputed_data = imputed_data.ffill()

# Label encoding
# List of categorical columns to encode
categorical_columns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
# Create an instance of LabelEncoder
lab_enc = LabelEncoder()
# Apply Label Encoding to each categorical column using a for loop
for column in categorical_columns:
    imputed_data[column] = lab_enc.fit_transform(imputed_data[column])

# Naive Bayes without oversampling
X_train_data = imputed_data.drop('income', axis=1)  # Features
y_train_data = imputed_data['income']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size=0.2, random_state=42)

# Apply RandomOverSampler to the training data
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Train Naive Bayes on the resampled data
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_resampled, y_resampled)

# Make predictions on the testing set
y_pred = naive_bayes_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_train = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report (Impute Missing Values Using Forward Fill):")
print(classification_report_train)

