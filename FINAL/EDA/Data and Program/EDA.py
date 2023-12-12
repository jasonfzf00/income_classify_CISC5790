import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

# Load training data
train_data = pd.read_csv('census-income.data.csv', header=None, delimiter=' ', names=column_names)
# Process first 14 columns
columns_to_process = train_data.columns[:14]
# Remove trailing commas in the first 14 columns
train_data[columns_to_process] = train_data[columns_to_process].apply(lambda x: x.str.replace(',', ''))

# Columns to visualize
columns_to_visualize = ['relationship', 'race', 'sex',]

# Set up separate plots for each column
for column in columns_to_visualize:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    fig.suptitle(f'Distribution of {column}')

    # Plot the distribution of the column by itself
    sns.histplot(train_data[column], kde=True, ax=axes[0])
    axes[0].set_title(f'{column} Distribution')

    # Plot the distribution of the column in relation to income
    sns.countplot(x=column, hue='income', data=train_data, ax=axes[1])
    axes[1].set_title(f'{column} Distribution by Income')

    # Rotate x-axis labels
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# For HPW
# Convert 'hours-per-week' to numeric
train_data['hours-per-week'] = pd.to_numeric(train_data['hours-per-week'], errors='coerce')

# Create bins for 'hours-per-week'
bins = [0, 20, 40, 60, 80, 100]
labels = ['0-20', '21-40', '41-60', '61-80', '81-100']

# Apply binning
train_data['hours-per-week_group'] = pd.cut(train_data['hours-per-week'], bins=bins, labels=labels, right=False)

# Set up a single plot for 'hours-per-week'
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Box Plot of Hours per Week by Income')

# Create a box plot of 'hours-per-week' in relation to income
sns.boxplot(x='income', y='hours-per-week', data=train_data, ax=ax)
ax.set_title('Hours per Week Box Plot by Income')

plt.tight_layout()
plt.show()


