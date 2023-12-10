import pandas as pd
import numpy as np

# PRE-PROCESSING
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

train_data = pd.read_csv('census-income.data.csv', header=None, delimiter=' ', names=column_names)

# Clean the data
columns_to_process = train_data.columns[:14]
# Remove trailing commas in the first 14 columns
train_data[columns_to_process] = train_data[columns_to_process].apply(lambda x: x.str.replace(',', ''))
# Turn ? into NaN
train_data.replace('?', np.nan, inplace=True)
# Turn <=50K: 0, >50K: 1
train_data['income'] = train_data['income'].replace({'<=50K': 0, '>50K': 1})
# Remove education column because it is redundant (we will keep education-num instead)
train_data.drop('education', axis=1, inplace=True)
# print(train_data.info())

# Forward fill missing data
imputed_data = train_data.copy()
imputed_data = imputed_data.ffill()


# NAIVE BAYES WITH K-FOLD CV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

# Split the data
X = imputed_data.drop('income', axis=1)  # Features
y = imputed_data['income']  # Target variable

# Separate categorical and numerical columns
categorical_columns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]),
                         columns=encoder.get_feature_names_out(categorical_columns))
X_encoded[numerical_columns] = X[numerical_columns]

# Initialize the Naive Bayes model
naive_bayes_model = GaussianNB()

# Initialize K-Folds
k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)

# Perform k-fold cross-validation
accuracy_scores = cross_val_score(naive_bayes_model, X_encoded, y, cv=kfold, n_jobs=1)

# Print individual accuracy scores
for idx, accuracy in enumerate(accuracy_scores):
    print(f"Accuracy for Fold {idx + 1}: {accuracy}")

# Calculate and print average accuracy
average_accuracy = np.mean(accuracy_scores)
print(f"\nAverage Accuracy: {average_accuracy}")
