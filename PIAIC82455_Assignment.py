#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


x = np.zeros(10)
x


# 3. Create a vector with values ranging from 10 to 49

# In[3]:


y = np.arange(10,49)
y


# 4. Find the shape of previous array in question 3

# In[4]:


np.shape(y)


# 5. Print the type of the previous array in question 3

# In[5]:


type(y)


# 6. Print the numpy version and the configuration
# 

# In[6]:


print(np.version)
print(np.show_config)


# 7. Print the dimension of the array in question 3
# 

# In[7]:


y.ndim


# 8. Create a boolean array with all the True values

# In[8]:


bool_array = [True,True,True,True,True,True]


# 9. Create a two dimensional array
# 
# 
# 

# In[9]:


x = np.array([[2,3,4],[5,6,7]])
x


# 10. Create a three dimensional array
# 
# 

# In[10]:


y = np.array([[[2,3,4],[3,4,5],[3,4,7]]])
y


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[11]:


x = [1,2,3,4,5,6,7,8]
x[-1:-9:-1]


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[12]:


y = np.zeros(10)
y[4]=1
y


# 13. Create a 3x3 identity matrix

# In[13]:


y = np.identity(3)
y


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[14]:


arr = np.array([1, 2, 3, 4, 5], dtype=float)
arr.dtype
arr


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[15]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])

x = arr1 * arr2
x


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[16]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])

x = arr1 == arr2
x


# 17. Extract all odd numbers from arr with values(0-9)

# In[17]:


a = np.array([0,1,2,3,4,5,6,7,8,9])
a[a%2 == 1]


# 18. Replace all odd numbers to -1 from previous array

# In[18]:


a = np.array([0,1,2,3,4,5,6,7,8,9])
a[a%2 == 1] = -1
a


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[19]:


arr = np.arange(10)
arr[[5,6,7,8]] = 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[20]:


arr = np.ones((5,5))
arr[1:-1, 1:-1] = 0
arr


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[21]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])

arr2d[1,1] = 12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[22]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

arr3d[0:1, 0:1] = 64

print(arr3d)


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[23]:


arr = np.arange(9).reshape(3,3)
arr2 = arr[0:1 , :]
print(arr2)
arr


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[24]:


arr = np.arange(9).reshape(3,3)
print(arr)

arr2 = arr[1][1]
arr2


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[25]:


arr = np.arange(9).reshape(3,3)
print(arr)

arr2 = arr[0:2 , 2:]
arr2


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[26]:


x = np.random.rand(10,10)
x
print(np.min(x))
print(np.max(x))


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[27]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

np.intersect1d(a,b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[28]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

z = a == b
z


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[29]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data[names != "Will"]


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[30]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)

data[names != "Will" &"Joe"]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[38]:


x = np.arange(1,16.0).reshape(5,3)
x


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[40]:


x = np.arange(1,17).reshape(2,2,4)
x


# 33. Swap axes of the array you created in Question 32

# In[44]:


y = np.swapaxes(x,0,1)
y


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[51]:


x = np.arange(10)
y = np.sqrt(x)

print(np.where(y<0.5,0,y))
y


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[58]:


a = np.random.rand(12)
print(a)
b = np.random.rand(12)
print(b)

np.maximum(a,b)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[60]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[62]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])

np.setdiff1d(a,b)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[66]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray

newColumn = np.array([[10,10,10]])

sampleArray[1,:] = newColumn

sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[67]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])

np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[68]:


x = np.random.rand(20)
print(x)

np.cumsum(x)


# In[ ]:




