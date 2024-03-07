#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[6]:


#load datset
iris_data = pd.read_csv("IRIS.csv")


# In[7]:


#Check for missing values
print("Number of missing values in each column:")
print(iris_data.isnull().sum())


# In[8]:


# Check distribution of target variable
print("\nDistribution of target variable:")
print(iris_data['species'].value_counts())


# In[9]:


# Convert species column to numerical labels
le = LabelEncoder()
iris_data['species'] = le.fit_transform(iris_data['species'])


# In[10]:


# Split features and target variable
X = iris_data.drop('species', axis=1)
y = iris_data['species']


# In[11]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[13]:


# Define a function to create the neural network model
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[15]:


# Define hyperparameters to search
batch_sizes = [16, 32, 64]
epochs_values = [50, 100, 150]
best_accuracy = 0
best_params = {}


# In[16]:


# Perform manual grid search
for batch_size in batch_sizes:
    for epochs_val in epochs_values:
        print(f"\nTraining model with batch size: {batch_size}, epochs: {epochs_val}")
        model = create_model()
        history = model.fit(X_train, y_train, epochs=epochs_val, batch_size=batch_size, 
                            validation_split=0.2, verbose=0, callbacks=[EarlyStopping(patience=5)])
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print("Test Accuracy:", accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'batch_size': batch_size, 'epochs': epochs_val}


# In[17]:


# Train the model with the best parameters
print("\nBest Parameters:", best_params)
model = create_model()
history = model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], 
                    validation_split=0.2, verbose=1, callbacks=[EarlyStopping(patience=5)])


# In[18]:


# Evaluate the best model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nTest Accuracy:", accuracy)


# In[19]:


# Make predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)


# In[20]:


# Generate classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted_labels))


# In[ ]:




