#!/usr/bin/env python
# coding: utf-8

# In[50]:


#Implementation of KNN
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# In[51]:


df = pd.read_csv ("census-income.data.csv", index_col = None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', 'capital-gain', 'capital-loss', "hours-per-week", "native-country", 'income'], sep=',\s',na_values=["?"],engine="python")


# In[52]:


df.head(30)


# In[53]:


df['income']


# In[54]:


missing_values = df.isnull().sum()
print("missing_values:")
print(missing_values)

df = df.dropna()


# In[55]:


missing_values = df.isnull().sum()
print("missing_values:")
print(missing_values)


# In[56]:


#Encoding variables
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


print(df['income'].value_counts())
#  # <=50K    24720
#  # >50K      7841

#Encoding for label. If <=50k then 0 else 1
label_info = {'<=50K':0, '>50K':1}
df['income'] = df['income'].map(label_info)
# # source:https://stackoverflow.com/questions/65716571/encoding-column-pandas-using-if-condition

# # after encoding 
print(f"After encoding", df['income'].value_counts())


# In[57]:


#Selecting features and splitting the datasets into features and label
selected_features = ['age', 'workclass', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', "hours-per-week", "native-country"]

x = df[selected_features] #feature
y = df.income #label


# In[58]:


#Splitting the data into test and train

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state=7)


# In[72]:


#Source: https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/
#implementing the KNN algorithm
#Assigning the nearest neighbor to 15 based on results below

model1 = KNeighborsClassifier(n_neighbors=15)

model1.fit(x_train, y_train)

#Predicitng the model in the test 
y_pred = model1.predict(x_test)


# In[60]:


#To check the performance of the model

print(model1.score(x_test, y_test))


# In[73]:


#Exporting prediction results to csv
prediction = pd.DataFrame(y_pred, columns=['KNN_predictions'])

prediction.to_csv('KNNPredictionTrain.csv', index= False)


# In[61]:


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




# In[62]:


#Now plotting the data
plt.plot(neighbor, te_acc, label = "Test data accuracy")
plt.plot(neighbor, tr_acc, label = "Train data accuracy")

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.show()


# In[34]:


##Should probably take 15 as the nearest neighbor. 


# In[63]:


#now checking from the test data
test = pd.read_csv ("census-income.test.csv", index_col = None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', 'capital-gain', 'capital-loss', "hours-per-week", "native-country", 'income'])

test[test==' ']= np.nan

test.head(15)



# In[64]:


missing_values = test.isnull().sum()
print("missing_values:")
print(missing_values)

test = test.dropna()


# In[65]:


#Checking if the missing values were dropped
missing_values = test.isnull().sum()
print("missing_values:")
print(missing_values)


# In[66]:


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

test.head()

#? got converted to zero


# In[26]:


test['income'].unique()


# In[67]:


print(test['income'].value_counts())

#Encoding for label. If <=50k then 0 else 1
label_info = {' <=50K.':0, ' >50K.':1}
test['income'] = test['income'].map(label_info)
# # source:https://stackoverflow.com/questions/65716571/encoding-column-pandas-using-if-condition

# # after encoding 
print(f"After encoding", test['income'].value_counts())


# In[75]:


selected_features = ['age', 'workclass', 'education-num','marital-status', 'occupation','relationship', 'race', 'sex', "hours-per-week", "native-country"]

#from the real test data
x_test_real = test[selected_features]
y_test_real = test.income

#Source: https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/
#implementing the KNN algorithm

#Predicitng the model in the test 
print(model1.predict(x_test_real))



# In[76]:


#To check the performance of the model

print(model1.score(x_test_real, y_test_real))


# In[77]:


#Exporting prediction results to csv
prediction = pd.DataFrame(y_pred, columns=['KNN_predictions'])

prediction.to_csv('KNNPredictionTest.csv', index= False)

