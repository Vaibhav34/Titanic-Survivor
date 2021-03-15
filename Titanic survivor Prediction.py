#!/usr/bin/env python
# coding: utf-8

# In[570]:


# importing lib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[571]:


#importing files

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[572]:


# printing shapes (rows n columns)

print(train.shape)
print(test.shape)


# In[573]:


train.info()


# In[574]:


test.info()


# In[575]:


train.drop(columns=['Cabin'],inplace=True)
test.drop(columns=['Cabin'],inplace=True)


# In[576]:


train['Embarked'].value_counts()


# In[577]:


train['Embarked'].fillna('S',inplace=True)


# In[578]:


test['Fare'].fillna(test['Fare'].mean(),inplace=True)


# In[579]:


train.isnull().sum()


# In[580]:


gen_age=np.random.randint(train['Age'].mean()-train['Age'].std(),train['Age'].mean()+train['Age'].std(),size=177)


# In[581]:


train['Age'][np.isnan(train['Age'])]=gen_age


# In[582]:


train.isnull().sum()


# In[583]:


gen_age1=np.random.randint(test['Age'].mean()-test['Age'].std(),test['Age'].mean()+test['Age'].std(),size=86)


# In[584]:


test['Age'][np.isnan(test['Age'])]=gen_age1


# In[585]:


test.isnull().sum()


# In[586]:


train[['Pclass','Survived']].groupby('Pclass').mean()


# In[587]:


train[['Sex','Survived']].groupby('Sex').mean()


# In[588]:


train[['Embarked','Survived']].groupby('Embarked').mean()


# In[589]:


sns.distplot(train['Age'])


# In[590]:


sns.boxplot(train['Age'])


# In[591]:


train[train['Age']>75]['Survived'].value_counts()


# In[592]:


sns.distplot(train[train['Survived']==0]['Age']) #dead graph


# In[593]:


sns.distplot(train[train['Survived']==1]['Age'])   #survived graph


# In[594]:


plt.subplots(figsize=(15,4))
sns.distplot(train[train['Survived']==0]['Age'])
sns.distplot(train[train['Survived']==1]['Age'])


# In[595]:


passengerId=test['PassengerId'].values


# In[596]:


train.drop(columns=['Ticket','PassengerId'],inplace=True)


# In[597]:


test.drop(columns=['Ticket','PassengerId'],inplace=True)


# In[598]:


sns.distplot(train['Fare'])


# In[599]:


sns.boxplot(train['Fare'])


# In[600]:


train[train['Fare']>400]['Survived'].value_counts()


# In[601]:


plt.subplots(figsize=(15,5))
sns.distplot(train[train['Survived']==0]['Fare'])
sns.distplot(train[train['Survived']==1]['Fare'])


# In[602]:


train['Name']


# In[603]:


train.drop(columns=['Name'],inplace=True)
test.drop(columns=['Name'],inplace=True)


# In[604]:


train['family']=train['SibSp']+train['Parch']+1
test['family']=test['SibSp']+test['Parch']+1


# In[605]:


train.drop(columns=['SibSp','Parch'],inplace=True)
test.drop(columns=['SibSp','Parch'],inplace=True)


# In[606]:


train.isnull().sum()


# In[607]:


plt.subplots(figsize=(15,5))
sns.distplot(train[train['Survived']==0]['Fare'])
sns.distplot(train[train['Survived']==1]['Fare'])


# In[608]:


train[['family','Survived']].groupby('family').mean()


# In[609]:


def family_size(number):
    if number==1:
        return "Alone"
    elif number>1 and number <5:
        return "small"
    else:
        return "large"
    


# In[610]:


family_size(5)


# In[611]:


train['family_size']=train['family'].apply(family_size)


# In[612]:


train.head()


# In[613]:


test['family_size']=test['family'].apply(family_size)


# In[614]:


test.head()


# In[615]:


train.drop(columns=['family'],inplace=True)
test.drop(columns=['family'],inplace=True)


# In[616]:


y=train['Survived'].values
y


# In[617]:


train.drop(columns=['Survived'],inplace=True)


# In[618]:


print(train.shape)
print(test.shape)


# In[619]:


train.append(test)


# In[620]:


final=train.append(test)
final.shape


# In[621]:


final


# In[622]:


pd.get_dummies(final,columns=['Pclass','Sex','Embarked','family_size'],drop_first=True)


# In[623]:


final=pd.get_dummies(final,columns=['Pclass','Sex','Embarked','family_size'],drop_first=True)


# In[624]:


final.shape


# In[625]:


xf=final.tail(418).values


# In[626]:


x=final.head(891).values


# In[627]:


y.shape


# In[628]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[629]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()


# In[630]:


clf.fit(x_train,y_train)


# In[631]:


y_pred=clf.predict(x_test)


# In[632]:


y_pred.shape


# In[633]:


y_test.shape


# In[634]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[635]:


yf=clf.predict(xf)


# In[636]:


yf.shape


# In[637]:


submission=pd.DataFrame()


# In[638]:


submission['PassengerId']=passengerId
submission['Survived']=yf


# In[639]:


submission.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




