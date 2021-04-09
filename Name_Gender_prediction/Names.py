#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score


# In[3]:


#read the data
df=pd.read_csv('allnames.tsv',sep='\t')

#droping columns with no use
df=df.drop('Person ID',1)

#changing the type of Gender
dic={'Male':0, 'Female':1}
df['Gender']=df['Gender'].map(dic)

# Gettin insights from mean, max, min, std
print(df.describe())

# see if there is any nan value
df.isna().any()


# In[4]:


# Getting data to start c
OutcomeColumn='Gender'
X=df.drop(OutcomeColumn,1)
y=df[OutcomeColumn]

X_train=X[df['Train/Test']=='Train']
y_train=y[df['Train/Test']=='Train']
X_test=X[df['Train/Test']=='Test']
y_test=y[df['Train/Test']=='Test']
# Creating traning set and test set


# In[5]:


import nltk
def gettingFirstName(name):
        fname=nltk.word_tokenize(name)[0]
        return fname
def gettingiLetter(name,i):
    return name[:i]

X_train["FirstName"] = X_train["Person Name"].apply(gettingFirstName)
X_test["FirstName"] = X_test["Person Name"].apply(gettingFirstName)
X_train["3Letter"] = X_train["FirstName"].apply(gettingiLetter,i=3)
X_test["3Letter"] = X_test["FirstName"].apply(gettingiLetter,i=3)
X_train["2Letter"] = X_train["FirstName"].apply(gettingiLetter,i=2)
X_test["2Letter"] = X_test["FirstName"].apply(gettingiLetter,i=2)


# In[6]:


# we use crosstab to get info about name to 0,1 mapping
ct_FirstName=pd.crosstab(X_train['FirstName'],y_train)
ct_3Letter=pd.crosstab(X_train['3Letter'],y_train)
ct_2Letter=pd.crosstab(X_train['2Letter'],y_train)


# In[7]:


# This method helps to find closest String
def getClosestString(name):
    close=name
    dis=5
    for temp in AllCTKeys:                             
        tempdis=nltk.edit_distance(temp, name)
        if tempdis<dis:
            close=temp
            dis=tempdis
        if dis==1:
            break
    return close


# In[8]:


from sklearn.metrics import classification_report
#Saving the keys for checks
AllCTKeys=set(ct_FirstName[0].index)
AllCt3Keys=set(ct_3Letter[0].index)
AllCt2Keys=set(ct_2Letter[0].index)

# This method will return majority of one of female count or male count from the name 
def checkFirstName(name):
    maleCount=ct_FirstName[0][name]
    femaleCount=ct_FirstName[1][name]
    if femaleCount>=maleCount:
        return 1
    else:
        return 0
# This method will return majority of one of female count or male count from 2 gram and 3 gram of chars
def checkNLetter(name,length,ct_NLetter):
    maleCount=ct_NLetter[0][name[:length]]
    femaleCount=ct_NLetter[1][name[:length]]
    if femaleCount>=maleCount:
        return 1
    else:
        return 0
    
# The model Works similar to a disicion tree, and also applying a probablity compersion on top
""" To Do
    Should apply some prebuilt models from sklearn to compare the accuracy and results""" 
def model(test):
    name=test["FirstName"]
# The algorithm first checks if the firstName is in the key set of training then getting max
    if name in AllCTKeys :
        return checkFirstName(name)
# If none of the above worked we check the closest string to find the answer 
    if name[:3] in AllCt3Keys:
        temp = getClosestString(name)
        if temp!=name:
            return checkFirstName(temp)
        else: 
            return checkNLetter(name,3,ct_3Letter)
    else:
        temp = getClosestString(name)
        if temp!=name:
            return checkFirstName(temp)
# If none of the above worked we check the first2 letters   
    if name[:2] in AllCt2Keys:
        return checkNLetter(name,2,ct_2Letter)
# If none of the above worked we return 1   
    return 1
    
y_pred=  X_test.apply(model,axis=1) 
print(classification_report(y_test, y_pred, target_names=['0','1']))
totalCorrect=len(y_pred[y_pred==y_test])
print(f'% of totalCorrect: {totalCorrect/len(y_pred)}')
print(f'totalCorrect:{totalCorrect}')

