#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import fashion_mnist

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data to include the channel dimension
x_train = x_train[..., None]
x_test = x_test[..., None]

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Input layer with image size 28x28 and 1 channel
    layers.Conv2D(32, (3, 3), activation='relu'),  # First Conv layer
    layers.MaxPooling2D(),  # Max pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Second Conv layer
    layers.MaxPooling2D(),  # Max pooling layer
    layers.Flatten(),  # Flatten the feature map
    layers.Dense(128, activation='relu'),  # Fully connected layer
    # Uncomment below for regularization if needed
    # layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(1e-4)),
    # layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    # layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes (Fashion MNIST)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# Print test accuracy
print(f"Test Accuracy: {test_acc:.4f}")


# In[ ]:




