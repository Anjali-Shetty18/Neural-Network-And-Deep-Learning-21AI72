#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tenserflow')


# In[2]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Function to create a neural network model
def create_model(activation_func):
    model = Sequential([
        Flatten(input_shape=(28, 28)),    # Flatten the 28x28 images
        Dense(128, activation=activation_func),
        Dense(64, activation=activation_func),
        Dense(10, activation='softmax')  # Softmax for output layer
    ])
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Function to train the model and plot accuracy
def train_and_evaluate(activation_func):
    print(f"Training with {activation_func} activation function")
    model = create_model(activation_func)
    
    history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test), verbose=0)
    
    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title(f'Accuracy with {activation_func} activation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Train and evaluate models with different activation functions
activations = ['sigmoid', 'tanh', 'relu']
for act in activations:
    train_and_evaluate(act)


# In[ ]:




