
# coding: utf-8

# In[33]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# ## Import the dataset in the attached CSV file

# In[28]:


collection = pd.read_csv("Movie_collection_train.csv")


# In[31]:


collection.head()


# ## Do EDD of the movie dataset and list down you observations.

# In[7]:


collection.describe()


# In[11]:


collection.Time_taken.fillna(value=collection.Time_taken.mean(), inplace=True)


# In[12]:


collection.describe()


# In[22]:


fig = plt.figure()


# In[42]:


sns.jointplot(x="Budget", y='Marketin_expense', data=collection)


# In[30]:


collection.index


# ### log transformation
# 

# In[37]:


new_budget = np.log1p(collection.Budget)


# In[38]:


new_budget


# In[40]:


sns.jointplot(x=new_budget, y='Marketin_expense', data=collection)


# ## Twitter_hastags

# In[41]:


sns.jointplot(x='Twitter_hastags', y='Marketin_expense', data=collection)


# ## Num_multiplex

# In[43]:


sns.jointplot(x='Num_multiplex', y='Marketin_expense', data=collection)


# ## Time_taken

# In[44]:


sns.jointplot(x='Time_taken', y='Marketin_expense', data=collection)


# ## Trailer_views

# In[51]:


sns.jointplot(x='Trailer_views', y='Marketin_expense', data=collection)


# ## Movie_length

# In[52]:


sns.jointplot(x='Movie_length', y='Marketin_expense', data=collection)


# In[56]:


log_m_length = 1/(collection.Movie_length)


# In[57]:


sns.jointplot(x=log_m_length, y='Marketin_expense', data=collection)


# In[61]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = pd.DataFrame(collection.Movie_length)


# In[63]:


std_m_length = scaler.fit_transform(data)


# In[69]:


std_m_length1 = std_m_length.reshape(1, -1)[0]


# In[70]:


sns.jointplot(x=std_m_length1, y='Marketin_expense', data=collection)

