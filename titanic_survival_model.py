#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 


# # loading the given dataset

# In[2]:


df = pd.read_csv(r'C:\Gowtham\Finger tips\All Projects\Python + ML\ML Project - Logistic Regression Titanic Survival\Titanic Survival.csv')
df.head()


# In[3]:


df.columns


# In[4]:


df.describe()


# # Checking for the null value

# In[5]:


df.isnull().sum()


# # Handle Null Values

# In[6]:


df = df.dropna()


# In[7]:


df.isnull().sum()


# In[8]:


df.head()


# # Part-2: Working with Models

# 1) Create the target data and feature data where target data is survived
# 2) Split the data into Training and testing Set
# 3) Create a Logistic regression model for Target and feature data
# 4) Display the Confusion Matrix
# 5) Find the Accuracy Score
# 6) Find the Precision Score
# 7) Find the Recall Score
# 8) Find the F1 Score
# 9) Find the probability of testing data
# 10) Display ROC Curve and find the AUC score 

# In[9]:


X = df.drop(['Survived','Name','Ticket'],axis=1)
y = df.Survived


# In[10]:


X


# # Applying label-encoding on categorical data

# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


enc =LabelEncoder()


# In[13]:


X['Sex'] = enc.fit_transform(X['Sex'])


# In[14]:


X['Cabin'] = enc.fit_transform(X['Cabin'])
X['Embarked'] = enc.fit_transform(X['Embarked'])


# In[15]:


X


# # Spliting the data into Training and testing Set 

# In[16]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# # Creating a Logistic regression model for Target and feature data

# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


clf = LogisticRegression()
clf.fit(X_train,y_train)


# In[19]:


clf.score(X_test,y_test)


# In[20]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,f1_score


# In[21]:


y_pred = clf.predict(X_test)


# # Confusion Matrix

# In[22]:


confusion_matrix(y_pred,y_test)


# # Accuracy Score 

# In[23]:


accuracy_score(y_pred,y_test)


# # Precision Score 

# In[24]:


precision_score(y_pred,y_test)


# # Recall Score 

# In[25]:


recall_score(y_pred,y_test)


# # F1 Score 

# In[26]:


f1_score(y_pred,y_test)


# # Findinding the probability of testing data 

# In[27]:


y_prob = clf.predict_proba(X_test)


# In[28]:


y_prob


# # ROC Curve and find the AUC score 

# In[29]:


from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, threshold = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc=" +str(auc))
plt.legend(loc=4)
plt.xlabel("False Positive Rate")
plt.ylabel(" Positive Rate")

plt.show()

