#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

# Define the inputs for AND, OR, and XOR gates
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Corresponding outputs for each gate
y_and = np.array([0, 0, 0, 1])  # AND gate
y_or = np.array([0, 1, 1, 1])   # OR gate
y_xor = np.array([0, 1, 1, 0])  # XOR gate

# Initialize Perceptron model
perceptron = Perceptron(max_iter=1000, tol=1e-3)

# Helper function to fit and evaluate perceptron
def evaluate_perceptron(X, y, gate_type):
    # Fit the perceptron model
    perceptron.fit(X, y)
    
    # Make predictions
    y_pred = perceptron.predict(X)
    
    # Output result
    print(f"{gate_type} Gate Predictions: {y_pred}")
    print(f"{gate_type} Gate Accuracy: {np.mean(y_pred == y) * 100:.2f}%")
    print(f"Weights: {perceptron.coef_}, Bias: {perceptron.intercept_}\n")

# Evaluate AND gate
evaluate_perceptron(X, y_and, "AND")

# Evaluate OR gate
evaluate_perceptron(X, y_or, "OR")

# Evaluate XOR gate
evaluate_perceptron(X, y_xor, "XOR")

# Visualization for XOR problem
def plot_xor_decision_boundary(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', s=100)
    plt.title('XOR Gate - Perceptron Decision Boundary')

    # Create a mesh to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.show()

# Plot XOR decision boundary (since it's not linearly separable, it will fail)
plot_xor_decision_boundary(X, y_xor, perceptron)


# In[ ]:




