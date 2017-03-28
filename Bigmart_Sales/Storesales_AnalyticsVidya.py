
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os
get_ipython().magic(u'matplotlib inline')
os.chdir('/Users/Wizard/GitHub/Myprojects/')
os.getcwd()


# In[4]:

#load files
df1 = pd.read_csv('Big Mart/Data/train.csv')
df2 = pd.read_csv('Big Mart/Data/test.csv')
df = pd.DataFrame(df1, copy =True)
tf = pd.DataFrame(df2, copy =True)
test = [df,tf]


# In[5]:

#descriptive statistics
print df.shape
print '\n%s\n' %df.dtypes
print df.head()
print '\ndescribe\n'
print df.describe()


# In[6]:

#dfcolumns
print df.columns
#unique and nulls
print df.apply(lambda x: len(x.unique()))
print '\nNull values count\n'
print df.isnull().sum()


# In[7]:

for i in test:
    i.loc[i['Item_Weight'].isnull(), 'Item_Weight'] = round(i['Item_Weight'].mean(),0)
    i['Item_Weight'].isnull().sum()
    i['Item_Weight'].mean()


# In[8]:

print 'sum of individual unique values in a selected column'

for i in df[['Item_Fat_Content','Item_Type','Outlet_Location_Type','Outlet_Type','Outlet_Size']]:
    print '\n %s'%i
    print df[i].value_counts()


# In[9]:

from scipy.stats import mode
for i in test:
    a = i.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x:mode(x).mode[0]))    
    i.loc[i['Outlet_Size'].isnull(), 'Outlet_Size']= i.loc[i['Outlet_Size'].isnull(),'Outlet_Type'].apply(lambda x:a[x])


# In[10]:

#print df.isnull().sum()
print df.describe()
print '\n ----- \n'
print df.head()


# In[11]:

#Feature engineering
#1 Modify Item_Visibility
for i in test:
    i.loc[i['Item_Visibility']==0, 'Item_Visibility'] = i['Item_Visibility'].mean()


# In[12]:

# 2. Food types
for i in test:
    i['Item_Type_New'] = i['Item_Identifier'].apply(lambda x:x[0:2])
    i['Item_Type_New'] = i['Item_Type_New'].map({'DR':'Drinks', 'FD':'Food', 'NC':'Non-consummables'})
    #print np.unique(df['Item_Type_New'])


# In[13]:

# 3. Fat items 
for i in test:
    np.unique(i.Item_Fat_Content)
    i.Item_Fat_Content = i.Item_Fat_Content.replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
    i.loc[i['Item_Type_New']=='Non-consummables', 'Item_Fat_Content'] = 'Non-edible'


# In[14]:

#store operating year
for i in test:
    i['Outlet_Operating_Year'] = 2013-i['Outlet_Establishment_Year']


# In[15]:

#Step 5: Numerical and One-Hot Coding of Categorical variables
#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in test:
    i['Outlet'] = i['Outlet_Identifier']
    col = [u'Item_Fat_Content', u'Outlet_Size', u'Outlet_Location_Type',u'Outlet_Type',u'Item_Type_New', u'Outlet']
    for j in col:
        i[j]=le.fit_transform(i[j])


# In[16]:

#One Hot Coding:
col = [u'Item_Fat_Content', u'Outlet_Size', u'Outlet_Location_Type',u'Outlet_Type', u'Item_Type_New',u'Outlet']
df = pd.get_dummies(df, columns = col)
tf = pd.get_dummies(tf, columns = col)


# In[16]:

df.head()


# In[17]:

df.drop(['Outlet_Establishment_Year','Item_Type'],axis=1, inplace =True)
tf.drop(['Outlet_Establishment_Year','Item_Type'],axis=1, inplace =True)


# In[18]:

print df.shape
print df.dtypes


# In[46]:

print tf.shape
print df.dtypes


# In[19]:

#Export files as modified versions:
df.to_csv("train_modified.csv",index=False)
tf.to_csv("test_modified.csv",index=False)


# In[20]:

#Model building
#mean model - base model
forecast_sales = df['Item_Outlet_Sales'].mean()
forecast_sales
#create a df for submission
base1 = tf[['Item_Identifier','Outlet_Identifier']]
base1['forecast_sales'] = forecast_sales
base1.head()


# In[57]:

df.dtypes


# In[21]:

#create target, id, 
target = 'Item_Outlet_Sales'
Id = ['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics
def modelfit(alg, train, test, predictor, target, Id, filename):
    alg.fit(train[predictor], train[target])
    #training predictions
    train_predictions = alg.predict(train[predictor])
    #test predictions
    test[target] = alg.predict(test[predictor])
    #validationscore 
    cv_score = cross_validation.cross_val_score(alg, train[predictor], train[target], 
                                                       cv=10,scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    print "\nModel Report"
    print "RMSE %4f" %np.sqrt(metrics.mean_squared_error(train[target].values, train_predictions))
    print "cv score mean - %4f|std - %4f| min - %4f| max - %4f" %(np.mean(cv_score),np.std(cv_score),
                                                                np.min(cv_score),np.max(cv_score))
    
    #Export submission file:
    Id.append(target)
    submission = pd.DataFrame({ x: test[x] for x in Id})
    submission.to_csv(filename, index=False)


# In[22]:

from sklearn.linear_model import LinearRegression
predictor = [x for x in df.columns if x not in [target]+Id]


# In[24]:

LR = LinearRegression(normalize = True) 
modelfit(LR, df, tf, predictor, target, Id, 'lr.csv')


# In[25]:

pd.Series(LR.coef_, predictor).sort_values().plot(kind = 'bar', title='Model Coefficients')


# In[26]:

print LR.intercept_


# In[27]:

print LR.predict(df[predictor])
print df['Item_Outlet_Sales']

