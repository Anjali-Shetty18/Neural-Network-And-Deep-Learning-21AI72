#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install tensorflow')


# In[6]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the MNIST dataset (or you can load your own dataset)
(X, y), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data: flatten images and normalize pixel values
X = X.reshape(-1, 28 * 28) / 255.0
X_test = X_test.reshape(-1, 28 * 28) / 255.0

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(300, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer with 300 units
    Dense(100, activation='relu'),  # Hidden layer with 100 units
    Dense(10, activation='softmax')  # Output layer with 10 classes (for classification)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=50, validation_data=(X_val, y_val))  # Adjust epochs as needed

# Predict the classes of the test set
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test accuracy: {accuracy:.4f}")


# In[8]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, and y_test are already defined

# Build the model
model = Sequential([
    Dense(300, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer with 300 units
    Dense(100, activation='relu'),  # Hidden layer with 100 units
    Dense(10, activation='softmax')  # Output layer with 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=50)  # Adjust the number of epochs as needed

# Predict the classes of the test set
y_pred_probs = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred_probs, axis=1)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test accuracy: {accuracy:.4f}")


# In[ ]:




