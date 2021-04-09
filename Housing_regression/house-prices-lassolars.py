#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

train=pd.read_csv('train.csv')
test= pd.read_csv('test.csv')


# In[2]:



# data preprocessing
df=train
df_test=test

Y=df['SalePrice']
df.drop(['SalePrice'],axis=1,inplace=True)

X=pd.get_dummies(df)
df_test=pd.get_dummies(df_test)

column1=X.columns
column2=df_test.columns
diff=np.setdiff1d(column1, column2)
for column in diff:
    df_test[column]=0
imputer = KNNImputer(n_neighbors=1)
X=imputer.fit_transform(X)
df_test=imputer.transform(df_test)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
X=scaler.transform(X)
df_test=scaler.transform(df_test)


# In[3]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=0)
max_val=0
max_index=0
for i in range(100):
    model = linear_model.LassoLars(alpha=10*i)
    model.fit(X_train,y_train)
    r_sq = model.score(X_valid, y_valid)
    if r_sq>max_val:
        max_val=r_sq
        max_index=i*10
        print('coefficient of determination:',i, r_sq)
model = linear_model.LassoLars(alpha=max_index)
model.fit(X,Y)
model.coef_


# In[4]:


predictions = model.predict(df_test)
test['Id']=test['Id'].astype(str)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
print(my_submission)
my_submission.to_csv('submission.csv', index=False)

