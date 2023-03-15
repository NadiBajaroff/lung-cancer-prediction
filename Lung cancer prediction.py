#!/usr/bin/env python
# coding: utf-8

# # Lung cancer prediction project

# Code to read the dataset, replace categorical features, scale age in a 0-1 range, define X and y, and split the dataset into training and testing data.
# Executable code below ⬇

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('survey lung cancer.csv')
data.replace({1: 0, 2: 1}, inplace=True)
data['LUNG_CANCER'].replace({'YES': 1, 'NO': 0}, inplace=True)
data['GENDER'].replace({'M': 1, 'F': 0}, inplace=True)
scaler = MinMaxScaler(feature_range =(0, 1))
age = data['AGE'].values
data['AGE'] = scaler.fit_transform(age.reshape(-1, 1))
data.drop(data[data['AGE']<0.3].index, inplace=True)
y= data['LUNG_CANCER']
X=data.drop(['LUNG_CANCER'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=data.LUNG_CANCER)


# The best model found for this case was the following:

# In[6]:


rand_for_clas = RandomForestClassifier(max_depth=11, max_features=3, max_samples=0.8, n_estimators=100)
rand_for_clas.fit(X_train, y_train)
predictions = rand_for_clas.predict(X_test)
print(classification_report(y_test,predictions))


# ### Process done during project development

# Read and explore dataset:

# In[2]:


data = pd.read_csv('survey lung cancer.csv')
data.head()


# In[114]:


data.describe()


# Handle binary categorical features:

# In[3]:


data.replace({1: 0, 2: 1}, inplace=True)


# In[4]:


data.head()


# In[5]:


data['LUNG_CANCER'].replace({'YES': 1, 'NO': 0}, inplace=True)


# In[118]:


data.head()


# In[6]:


data['GENDER'].replace({'M': 1, 'F': 0}, inplace=True)


# In[7]:


scaler = MinMaxScaler(feature_range =(0, 1))
age = data['AGE'].values
data['AGE'] = scaler.fit_transform(age.reshape(-1, 1))
data.head()


# EDA:

# In[90]:


sns.countplot(data=data, x='LUNG_CANCER')


# In[121]:


corr = data.corr()
plt.figure(figsize=(12,7))
sns.heatmap(corr, annot=True)


# In[122]:


corr['LUNG_CANCER'].sort_values()


# In[123]:


plt.figure(figsize=(12,7))
sns.histplot(data=data, x='AGE', bins=15)


# In[124]:


sns.countplot(data=data, x='SMOKING')


# In[14]:


corr = data.corr()
corr['LUNG_CANCER'].sort_values()


# Feature engineering to explore age deeply:

# In[15]:


def age_3_bins(age):
    if age <51:
        return 0
    elif age >50 and age <65:
        return 1
    else:
        return 2


# In[16]:


data['edad_3_bins'] = data['AGE'].apply(age_3_bins)


# In[17]:


def age_7_bins(age):
    if age <31:
        return 0
    elif age >30 and age <41:
        return 1
    elif age >40 and age <51:
        return 2
    elif age >50 and age <61:
        return 3
    elif age >60 and age <71:
        return 4
    elif age >70 and age <81:
        return 5
    else:
        return 6


# In[18]:


data['edad_7_bins'] = data['AGE'].apply(age_7_bins)
data.head()


# In[125]:


corr = data.corr()
corr['LUNG_CANCER'].sort_values()


# In[20]:


def older(age):    
    return 65 if age >64 else age


# In[21]:


data['AGE'] = data['AGE'].apply(older)
data.head()


# In[126]:


corr = data.corr()
corr['LUNG_CANCER'].sort_values()


# EDA:

# In[23]:


sns.countplot(data=data, x='SMOKING', hue='LUNG_CANCER', palette='rainbow')


# In[24]:


sns.countplot(data=data, x='SHORTNESS OF BREATH', hue='LUNG_CANCER', palette='rainbow')


# In[25]:


sns.countplot(data=data, x='GENDER', hue='LUNG_CANCER', palette='rainbow')


# In[26]:


sns.countplot(data['LUNG_CANCER'])


# In[28]:


fig, axs = plt.subplots(4, 4, figsize=(20, 20))
axs = axs.ravel()

features = data.drop(['LUNG_CANCER', 'AGE'], axis=1)

for i, col in enumerate(features):
    ax = sns.countplot(x=col, data=data, hue='LUNG_CANCER', ax=axs[i], palette='pastel')
    ax.legend().remove()

# Remove the x-tick labels for all subplots except the bottom row
for ax in axs[-2:]:
    ax.set_xticklabels([])

plt.tight_layout()
plt.show()


# In[29]:


sns.violinplot(x='LUNG_CANCER', y='edad_3_bins', data=data, palette=['pink', 'brown'])


# In[30]:


sns.violinplot(x='LUNG_CANCER', y='edad_7_bins', data=data, palette=['pink', 'brown'])


# In[31]:


fig, axs = plt.subplots(4, 4, figsize=(20, 20))
axs = axs.ravel()

features = data.drop(['edad_7_bins', 'edad_3_bins', 'AGE'], axis=1)

for i, col in enumerate(features):
    ax = sns.violinplot(x=col, y='edad_7_bins', data=data, ax=axs[i], palette=['pink', 'brown'])
    #ax.legend().remove()

# Remove the x-tick labels for all subplots except the bottom row
for ax in axs[-2:]:
    ax.set_xticklabels([])

plt.tight_layout()
plt.show()


# Testing different models. For this case the best metric to consider is recall, since the most important thing is to maximize true positives and minimize false negatives

# In[12]:


y= data['LUNG_CANCER']
X=data.drop(['LUNG_CANCER'], axis=1)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# #### Logistic Regression 

# In[14]:


logreg = LogisticRegression()


# In[15]:


logreg.fit(X_train, y_train)


# In[16]:


predictions = logreg.predict(X_test)


# In[37]:


print(classification_report(y_test, predictions))


# In[38]:


sns.countplot(x='SMOKING', data=data, hue='PEER_PRESSURE', palette='pastel')


# In[17]:


## data stratification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=data.LUNG_CANCER)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)
print(classification_report(y_test, predictions))


# In[40]:


X_train_older = X_train[X_train.edad_7_bins > 1]


# In[10]:


X_train_older, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=data.LUNG_CANCER)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)
print(classification_report(y_test, predictions))


# #### Decision trees and Random forests 

# In[18]:


dtree = DecisionTreeClassifier()
rfc = RandomForestClassifier()


# In[19]:


dtree.fit(X_train, y_train)
rfc.fit(X_train, y_train)


# In[20]:


predictionsDT = dtree.predict(X_test)
predictionsRF = rfc.predict(X_test)


# In[45]:


print(classification_report(y_test,predictionsDT))
print(classification_report(y_test,predictionsRF))


# In[46]:


# twicking the RFC model
rfc2 = RandomForestClassifier(n_estimators=200)
rfc2.fit(X_train, y_train)
predictionsRF2 = rfc2.predict(X_test)
print(classification_report(y_test,predictionsRF2))


# In[21]:


hyperparameters = {'n_estimators':[100,200,300,500,1000]}

rfc3 = RandomForestClassifier()
gs = GridSearchCV(estimator=rfc3, param_grid=hyperparameters, n_jobs=-1)


# In[22]:


gs.fit(X_train, y_train)


# In[23]:


gs.best_params_


# In[50]:


gs.score(X_test, y_test)


# In[141]:


rfc4 = RandomForestClassifier(n_estimators=300)
rfc4.fit(X_train, y_train)
predictionsRF4 = rfc4.predict(X_test)
print(classification_report(y_test,predictionsRF4))


# In[6]:


hyperparameters = {'n_estimators':[100,200,300,500,1000],
                   'max_samples':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                   'max_features':[1,2,3,4,5,6,7],
                   'max_depth':[None,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}

rfc5 = RandomForestClassifier()
gs = GridSearchCV(estimator=rfc5, param_grid=hyperparameters, n_jobs=-1)

gs.fit(X_train, y_train)
print(gs.best_params_)
gs.score(X_test, y_test)


# In[5]:


rfc6 = RandomForestClassifier(max_depth=11, max_features=3, max_samples=0.8, n_estimators=100)
rfc6.fit(X_train, y_train)
predictionsRF6 = rfc6.predict(X_test)
print(classification_report(y_test,predictionsRF6))


# In[144]:


hyperparameters = {'max_features':[1,2,3,4,5,6,7]}

rfc7 = RandomForestClassifier()
gs = GridSearchCV(estimator=rfc7, param_grid=hyperparameters, n_jobs=-1)

gs.fit(X_train, y_train)
print(gs.best_params_)
gs.score(X_test, y_test)


# In[145]:


rfc8 = RandomForestClassifier(n_estimators=300, max_samples=0.6, max_features=4)
rfc8.fit(X_train, y_train)
predictionsRF8 = rfc8.predict(X_test)
print(classification_report(y_test,predictionsRF8))


# In[148]:


hyperparameters = {'max_depth':[None,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}

rfc9 = RandomForestClassifier()
gs = GridSearchCV(estimator=rfc9, param_grid=hyperparameters, n_jobs=-1)

gs.fit(X_train, y_train)
print(gs.best_params_)
gs.score(X_test, y_test)


# In[151]:


rfc10 = RandomForestClassifier(n_estimators=300, max_samples=0.6, max_features=4, max_depth=20)
rfc10.fit(X_train, y_train)
predictionsRF10 = rfc10.predict(X_test)
print(classification_report(y_test,predictionsRF10))


# #### XGBoost 

# In[152]:


XGB = XGBClassifier()
XGB.fit(X_train, y_train)
predictionsXGB = XGB.predict(X_test)
print(classification_report(y_test,predictionsXGB))


# #### Support vector machines 

# In[132]:


SVC = SVC()
SVC.fit(X_train, y_train)
predictionsSVC = SVC.predict(X_test)
print(classification_report(y_test,predictionsSVC))


# In[ ]:




