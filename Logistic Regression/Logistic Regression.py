#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler 


# In[4]:


#Importing the income dataset

df = pd.read_csv ("census-income.data.csv", index_col = None)

#Previewing the data
df.head()


# In[6]:


#Selecting features and splitting the datasets into features and label
selected_features = ['age', 'workclass', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', "hours-per-week", "native-country"]

x = df[selected_features] #feature
y = df.income #label

#Balancing the data
ros = RandomOverSampler()
ros.fit(x, y)

X, Y = ros.fit_resample(x, y)


# In[7]:


#Splitting the dataset

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state=7)


# In[8]:


#Logistic Regression
#Source: https://www.datacamp.com/tutorial/understanding-logistic-regression-python?irclickid=XPG32xUYdxyPWbHWdp29132wUkFSTqSZwQmx3c0&irgwc=1&im_rewards=1&utm_medium=affiliate&utm_source=impact&utm_campaign=000000_1-2003851_2-mix_3-all_4-na_5-na_6-na_7-mp_8-affl-ip_9-na_10-bau_11-Bing%20Rebates%20by%20Microsoft&utm_content=BANNER&utm_term=EdgeBingFlow

#setting a seed
model1 = LogisticRegression(max_iter=1000,random_state=7)

#fitting the model using our data
model1.fit(x_train, y_train)

#predicting on the test data.
y_pred = model1.predict(x_test)


# In[9]:


print(y_pred)


# In[10]:


#Evaluation of model

#first using confusion matrix

from sklearn import metrics

confusion_mat = metrics.confusion_matrix(y_test, y_pred)
confusion_mat


# In[11]:


#Classification report
from sklearn.metrics import classification_report

income_label = ['<=50k', '>50k']

print(classification_report(y_test, y_pred, target_names=income_label))


# In[12]:


#ROC Curve
import matplotlib.pyplot as plt

y_pred_probability = model1.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_probability)
auc = metrics.roc_auc_score(y_test, y_pred_probability)
plt.plot(fpr, tpr, label = "data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[13]:


#now checking from the test data
test = pd.read_csv ("census-income.test.csv", index_col = None)
test.head()


# In[14]:


len(test)


# In[15]:


test.info()


# In[16]:


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


# In[17]:


#Exporting Logistic Regression Prediction results to csv
#Exporting prediction results to csv
prediction = pd.DataFrame(y_pred_real, columns=['logistic_predictions'])

prediction.to_csv('LogisticPredictionTest.csv', index= False)


# In[18]:


len(prediction)


# In[19]:


#Classification report
from sklearn.metrics import classification_report

income_label = ['<=50k', '>50k']

print(classification_report(y_test_real, y_pred_real, target_names=income_label))


# In[20]:


#ROC Curve
import matplotlib.pyplot as plt

y_pred_probability_real = model1.predict_proba(x_test_real)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test_real, y_pred_probability_real)
auc = metrics.roc_auc_score(y_test_real, y_pred_probability_real)
plt.plot(fpr, tpr, label = "data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[21]:


#! for notebook command, without ! it is cmd prompt command
get_ipython().system('jupyter nbconvert --to script *.ipynb')

