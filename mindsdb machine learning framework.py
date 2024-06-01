#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install mindsdb


# In[3]:


pip install mindsdb scikit-learn pandas


# In[4]:


import mindsdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


# In[5]:


# Load the dataset
loan = pd.read_csv("Downloads/Loan_default.csv")


# In[6]:


# Sample a subset if the dataset is too large (e.g., 10,000 rows)
loan = loan.sample(n=10000, random_state=42)


# In[7]:


loan.head()


# In[8]:


# Fill missing values
loan = loan.fillna(method='ffill')



# Display the cleaned data
print(loan.head())


# In[9]:


# Convert categorical variables to numerical using one-hot encoding
loan = pd.get_dummies(loan, dtype=int)


# In[10]:


import mindsdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[11]:


# Split the dataset into features (X) and target (y)
X = loan.drop(columns=['Default'])
y = loan['Default']


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[15]:


# Initialize MindsDB
mdb = mindsdb.Predictor(name='loan_default_predictor')

# Combine the features and target into a single DataFrame for training
train_data = pd.DataFrame(X_train, columns=data.drop(columns=['default_status']).columns)
train_data['default_status'] = y_train.values

# Train the model
mdb.learn(
    from_data=train_data,
    to_predict='default_status'
)

# Prepare new data for prediction
test_data = pd.DataFrame(X_test, columns=data.drop(columns=['default_status']).columns).iloc[:5]

# Make predictions
predictions = mdb.predict(when_data=test_data)

# Display the predictions
print(predictions)


# In[18]:


import mindsdb
mdb = mindsdb.Predictor()


# In[20]:


from mindsdb import MindsDB

# Initialize MindsDB
mdb = MindsDB()

# Train the model
mdb.learn(from_data='"Downloads/Loan_default.csv', to_predict='Default')

# Make predictions
predictions = mdb.predict(when_data='new_data_to_predict.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




