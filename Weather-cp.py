
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#pd.Series(np.random.randn(5),index=['a','b','c','d','e'])


# In[3]:


#s=pd.Series(np.random.randn(5))
#print(s)


# In[4]:


#df=pd.DataFrame(s,columns=['Coloumn 1'])
#df


# In[ ]:


import matplotlib.pyplot as plt

plt.style.use('ggplot')


# In[3]:


df=pd.read_csv('tas.csv')
df.tail()


# In[4]:


df=df.iloc[:,:2]
df.head()


# In[5]:


df.describe()


# In[7]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
plt.plot(df['tas'])
plt.title("Historical Temparature Data For Kenya 1901-2015")
plt.xlabel("Year")
plt.ylabel("Temparature")
plt.show()


# In[8]:


print(type(df['date'][1]))


# In[11]:



date=pd.to_datetime(df['date']).dt.year
grouped=df.groupby([date]).mean()
print(grouped)


# In[12]:


plt.figure(figsize=(15,5))
plt.plot(grouped['tas'])

plt.title("Historical Temparature Data For Kenya 1901-2015")
plt.xlabel("Year")
plt.ylabel("Temparature")
plt.show()


# In[18]:


df[date==1971]


# In[21]:


#Modelling
from sklearn.linear_model import LinearRegression as LinReg
x=grouped.index.values.reshape(-1,1)
y=grouped['tas'].values
reg=LinReg()
reg.fit(x,y)
y_preds = reg.predict(x)
print("Accuracy: "+str(reg.score(x,y)))
print("2020 Temparature Prediction: "+str(reg.predict(2020)))
print("2025 Temparature Prediction: "+str(reg.predict(2025)))
print("2030 Temparature Prediction: "+str(reg.predict(2030)))

# In[23]:


plt.figure(figsize=(15,5))
plt.title("Linear Regression")
plt.scatter(x=x,y=y_preds)
plt.scatter(x=x,y=y,c='r')


# In[30]:




