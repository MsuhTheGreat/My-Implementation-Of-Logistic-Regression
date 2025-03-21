import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Make necessary directories
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Fixed Variables
ALPHA = 0.00001
EPOCHS = 1_000_001


def load_data():
    """Load and split the breast cancer dataset."""
    # Loading breast cancer data from scikit-learn
    data = load_breast_cancer()
    X, Y = data.data, data.target
    # Splitting data into training data and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    
    # Transpose X so that features are rows and examples are columns
    X_train, X_test = X_train.T, X_test.T

    # Convert Y to row vectors
    Y_train, Y_test = Y_train.reshape(1, -1), Y_test.reshape(1, -1)

    return X_train, X_test, Y_train, Y_test


def save_data(X_train, X_test, Y_train, Y_test):
    """Save datasets for reuse."""
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "Y_train.npy"), Y_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "Y_test.npy"), Y_test)


def initialize_parameters(nx):
    """Initialize weights and bias."""
    W = np.random.randn(1, nx)
    b = np.random.randn(1, 1)
    return W, b


def train(X_train, Y_train, alpha=ALPHA, epochs=EPOCHS):
    """Train logistic regression model using gradient descent."""
    nx, m = X_train.shape
    epsilon = 1e-8
    W, b = initialize_parameters(nx)

    for i in range(epochs):
        # Forward Propagation
        Z = np.dot(W, X_train) + b
        Z = np.clip(Z, -500, 500)   # Avoid overflow
        A = 1 / (1 + np.exp(-Z))    # Sigmoid

        # Cost Function
        J = -np.mean(Y_train * np.log(A + epsilon) + (1 - Y_train) * np.log(1 - A + epsilon))

        # Backpropagation
        dZ = A - Y_train
        dW = np.dot(dZ, X_train.T) / m
        db = np.sum(dZ) / m

        # Update Parameters
        W -= alpha * dW
        b -= alpha * db * 0.1  # Bias decay for stability

        # Print status
        if i % 10_000 == 0:
            print(f"{i} epochs done. Cost: {J:.4f}")

    print("Final Weights:", W)
    print("Final Bias:", b)

    # Save Model
    np.save(os.path.join(MODEL_DIR, "W.npy"), W)
    np.save(os.path.join(MODEL_DIR, "b.npy"), b)
    np.save(os.path.join(MODEL_DIR, "J.npy"), J)


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_data()
    save_data(X_train, X_test, Y_train, Y_test)
    train(X_train, Y_train)
