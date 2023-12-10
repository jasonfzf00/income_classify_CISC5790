#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[43]:


#Importing the income dataset
df = pd.read_csv ("census-income.data.csv", index_col = None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', 'capital-gain', 'capital-loss', "hours-per-week", "native-country", 'income'], sep=',\s',na_values=["?"],engine="python")


# In[44]:


#Previewing the data
df.head(20)


# In[45]:


#Finding missing values (?) for each variables, manually, indiviudally. 
# df['age'].unique()
# df['workclass'].unique()
# df['fnlwgt'].unique()
# df['education'].unique()
# df['education-num'].unique()
# df['marital-status'].unique()
# df['occupation'].unique()
# df['relationship'].unique()
# df['race'].unique()
# df['sex'].unique()
# df['capital-gain'].unique()
# df['capital-loss'].unique()
# df['hours-per-week'].unique()
# df['native-country'].unique()
# df['income'].unique()

missing_values = df.isnull().sum()
print("missing_values:")
print(missing_values)

df = df.dropna()


# In[7]:


missing_values = df.isnull().sum()
print("missing_values:")
print(missing_values)


# In[8]:


#looking at the datatype
print(df.info())

#have to transform the variables for regression


# In[9]:


df['income'].unique()


# In[46]:


#encoding categorical variables to numeric codes
from sklearn.preprocessing import LabelEncoder

#calling labelencoder
lab_enc = LabelEncoder()

#performing label encoding in the select features
df['workclass'] = lab_enc.fit_transform(df['workclass'])
df['education'] = lab_enc.fit_transform(df['education'])
df['marital-status'] = lab_enc.fit_transform(df['marital-status'])
df['occupation'] = lab_enc.fit_transform(df['occupation'])
df['relationship'] = lab_enc.fit_transform(df['relationship'])
df['race'] = lab_enc.fit_transform(df['race'])
df['sex'] = lab_enc.fit_transform(df['sex'])
df['native-country'] = lab_enc.fit_transform(df['native-country'])


# In[47]:


print(df['income'].value_counts())
#  # <=50K    24720
#  # >50K      7841

#Encoding for label. If <=50k then 0 else 1
label_info = {'<=50K':0, '>50K':1}
df['income'] = df['income'].map(label_info)
# # source:https://stackoverflow.com/questions/65716571/encoding-column-pandas-using-if-condition

# # after encoding 
print(f"After encoding", df['income'].value_counts())


# In[12]:


df.head()


# In[48]:


#Selecting features and splitting the datasets into features and label
selected_features = ['age', 'workclass', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', "hours-per-week", "native-country"]

x = df[selected_features] #feature
y = df.income #label


# In[49]:


#Splitting the dataset

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state=7)


# In[74]:


#Logistic Regression
#Source: https://www.datacamp.com/tutorial/understanding-logistic-regression-python?irclickid=XPG32xUYdxyPWbHWdp29132wUkFSTqSZwQmx3c0&irgwc=1&im_rewards=1&utm_medium=affiliate&utm_source=impact&utm_campaign=000000_1-2003851_2-mix_3-all_4-na_5-na_6-na_7-mp_8-affl-ip_9-na_10-bau_11-Bing%20Rebates%20by%20Microsoft&utm_content=BANNER&utm_term=EdgeBingFlow

#setting a seed
model1 = LogisticRegression(max_iter=1000,random_state=7)

#fitting the model using our data
model1.fit(x_train, y_train)

#predicting on the test data.
y_pred = model1.predict(x_test)


# In[81]:


#Exporting prediction results to csv
prediction = pd.DataFrame(y_pred, columns=['logistic_predictions'])

prediction.to_csv('LogisticPredictionTrain.csv', index= False)


# In[51]:


print(y_pred)


# In[52]:


#Evaluation of model

#first using confusion matrix

from sklearn import metrics

confusion_mat = metrics.confusion_matrix(y_test, y_pred)
confusion_mat


# In[53]:


#Classification report
from sklearn.metrics import classification_report

income_label = ['<=50k', '>50k']

print(classification_report(y_test, y_pred, target_names=income_label))


# In[54]:


#ROC Curve
import matplotlib.pyplot as plt

y_pred_probability = model1.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_probability)
auc = metrics.roc_auc_score(y_test, y_pred_probability)
plt.plot(fpr, tpr, label = "data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[59]:


#now checking from the test data
test = pd.read_csv ("census-income.test.csv", index_col = None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', 'capital-gain', 'capital-loss', "hours-per-week", "native-country", 'income'])
test.head(30)


# In[60]:


#Finding missing values (?) for each variables, manually, indiviudally. 
# df['age'].unique()
# df['workclass'].unique()
# df['fnlwgt'].unique()
# df['education'].unique()
# df['education_num'].unique()
# df['marital_status'].unique()
# df['occupation'].unique()
# df['relationship'].unique()
# df['race'].unique()
# df['sex'].unique()
# df['capital_gain'].unique()
# df['capital_loss'].unique()
# df['hours_per_week'].unique()
# df['native_country'].unique()
# df['income'].unique()
#converting ? to nan
test[test==' ']= np.nan

test.head(15)


# In[61]:


test.info()


# In[62]:


##Dropping all the missing values
#Removing all missing values
test = test.dropna()
missing_values = test.isnull().sum()
print("missing_values:")
print(missing_values)


# In[63]:


test.head(30)


# In[64]:


#encoding categorical variables to numeric codes
from sklearn.preprocessing import LabelEncoder

#calling labelencoder
lab_enc = LabelEncoder()

#performing label encoding in the select features
test['workclass'] = lab_enc.fit_transform(test['workclass'])
test['education'] = lab_enc.fit_transform(test['education'])
test['marital-status'] = lab_enc.fit_transform(test['marital-status'])
test['occupation'] = lab_enc.fit_transform(test['occupation'])
test['relationship'] = lab_enc.fit_transform(test['relationship'])
test['race'] = lab_enc.fit_transform(test['race'])
test['sex'] = lab_enc.fit_transform(test['sex'])
test['native-country'] = lab_enc.fit_transform(test['native-country'])


# In[65]:


test.head()

#? got converted to zero


# In[66]:


test['income'].unique()


# In[67]:


print(test['income'].value_counts())

#Encoding for label. If <=50k then 0 else 1
label_info = {' <=50K.':0, ' >50K.':1}
test['income'] = test['income'].map(label_info)
# # source:https://stackoverflow.com/questions/65716571/encoding-column-pandas-using-if-condition

# # after encoding 
print(f"After encoding", test['income'].value_counts())


# In[68]:


test.head()


# In[82]:


selected_features = ['age', 'workclass', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', "hours-per-week", "native-country"]

#from the real test data
x_test_real = test[selected_features]
y_test_real = test.income

y_pred_real = model1.predict(x_test_real)


#Evaluation of model

#first using confusion matrix

from sklearn import metrics

confusion_mat = metrics.confusion_matrix(y_test_real, y_pred_real)
confusion_mat


# In[83]:


#Exporting Logistic Regression Prediction results to csv
#Exporting prediction results to csv
prediction = pd.DataFrame(y_pred, columns=['logistic_predictions'])

prediction.to_csv('LogisticPredictionTest.csv', index= False)


# In[70]:


#Classification report
from sklearn.metrics import classification_report

income_label = ['<=50k', '>50k']

print(classification_report(y_test_real, y_pred_real, target_names=income_label))


# In[71]:


#ROC Curve
import matplotlib.pyplot as plt

y_pred_probability_real = model1.predict_proba(x_test_real)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test_real, y_pred_probability_real)
auc = metrics.roc_auc_score(y_test_real, y_pred_probability_real)
plt.plot(fpr, tpr, label = "data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

