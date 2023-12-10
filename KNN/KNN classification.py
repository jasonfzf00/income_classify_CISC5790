#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Implementation of KNN
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler 


# In[2]:


df = pd.read_csv ("census-income.data.csv", index_col = None)


# In[6]:


df.head()


# In[4]:


#Selecting features and splitting the datasets into features and label
selected_features = ['age', 'workclass', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', "hours-per-week", "native-country"]

x = df[selected_features] #feature
y = df.income #label

#Balancing the data
ros = RandomOverSampler()
ros.fit(x, y)

X, Y = ros.fit_resample(x, y)


# In[5]:


#Splitting the data into test and train

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state=7)


# In[20]:


#Source: https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/
#implementing the KNN algorithm
#Assigning the nearest neighbor to 5 based on results below

model1 = KNeighborsClassifier(n_neighbors=3)

model1.fit(x_train, y_train)

#Predicitng the model in the test 
y_pred = model1.predict(x_test)


# In[21]:


#To check the performance of the model

print(model1.score(x_test, y_test))


# In[ ]:


#Deciding the k-value for the dataset. 
#Source: https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/

neighbor = np.arange(1, 30)

tr_acc = np.empty(len(neighbor)) #train accuracy
te_acc = np.empty(len(neighbor)) #test accuracy

#now looping throught the k values
for i, k in enumerate(neighbor):
    model2 = KNeighborsClassifier(n_neighbors=k)
    model2.fit(x_train, y_train)

#Computing the model accuracy
    tr_acc[i] = model2.score(x_train, y_train)
    te_acc[i] = model2.score(x_test, y_test)


# In[ ]:


#Now plotting the data
plt.plot(neighbor, te_acc, label = "Test data accuracy")
plt.plot(neighbor, tr_acc, label = "Train data accuracy")

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.show()


# In[34]:


##Should probably take 15 as the nearest neighbor. 


# In[15]:


#now checking from the test data
test = pd.read_csv ("census-income.test.csv", index_col = None)
test.head()


# In[21]:


selected_features = ['age', 'workclass', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', "hours-per-week", "native-country"]

#from the real test data
x_test_real = test[selected_features]
y_test_real = test.income

#Source: https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/
#implementing the KNN algorithm

#Predicitng the model in the test 
y_pred_real = model1.predict(x_test_real)
print(y_pred_real)


# In[22]:


#To check the performance of the model

print(model1.score(x_test_real, y_test_real))


# In[23]:


#Exporting prediction results to csv
prediction = pd.DataFrame(y_pred_real, columns=['KNN_predictions'])

prediction.to_csv('KNNPredictionTest.csv', index= False)


# In[24]:


len(prediction)


# In[78]:


get_ipython().system('jupyter nbconvert --to script *.ipynb')

