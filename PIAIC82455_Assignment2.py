#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[1]:


import numpy as np
x = np.arange(10)

x.reshape(2,5)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[2]:


x = np.arange(10).reshape(2,5)
y = np.ones(10, dtype=int).reshape(2,5)

print(x)
print(y)

np.vstack((x,y))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[3]:


print(x)
print(y)

np.hstack((x,y))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[4]:


x = np.arange(10).reshape(2,5)
np.ravel(x)


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[5]:


x 
np.ravel(x)


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[6]:


np.arange(15).reshape(5,3)


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[7]:


x = np.arange(25).reshape(5,5)
print(x)
np.power(x,2)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[8]:


x = np.arange(30).reshape(5,6)
print(x)

np.mean(x)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[9]:


np.std(x)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[10]:


np.median(x)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[11]:


np.transpose(x)


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[12]:


y = np.arange(16).reshape(4,4)
print(y)

np.trace(y)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[16]:


display(np.linalg.det(y))


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[18]:


x = np.percentile(y, 5)
z = np.percentile(y, 95)

print(x,z)


# ## Question:15

# ### How to find if a given array has any null values?

# In[24]:


np.isnan(y)


# In[ ]:




