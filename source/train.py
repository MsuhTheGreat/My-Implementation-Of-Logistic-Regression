import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os

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
    # Splitting data into training data and testing data and saving it
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    # Structuring Output Data
    Y_train, Y_test = Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)
    # Return the Input and Output Data
    return X_train, X_test, Y_train, Y_test


def save_data(X_train, X_test, Y_train, Y_test):
    """Save datasets for reuse."""
    # Saving Into Files to Reuse Again
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "Y_train.npy"), Y_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "Y_test.npy"), Y_test)


def initialize_parameters(nx):
    """Initialize weights and bias."""
    # Initialize weights and bias as normalized
    W = np.random.randn(nx, 1)
    b = np.random.randn(1, 1)
    return W, b


def train(X_train, Y_train, alpha=ALPHA, epochs=EPOCHS):
    """Train logistic regression model using gradient descent."""
    m, nx = X_train.shape
    epsilon = 1e-8
    W, b = initialize_parameters(nx)

    # Iterate through all m examples
    for i in range(epochs):
        # Calculating predicted values
        Z = np.dot(X_train, W) + b
        # Avoid overflowing
        Z = np.clip(Z, -500, 500)
        A = 1 / (1 + np.exp(-Z))

        # Calculate cost function
        J = -np.mean(Y_train * np.log(A + epsilon) + (1 - Y_train) * np.log(1 - A + epsilon))

        # Calculate necessary determinants
        dZ = A - Y_train
        dW = np.dot(X_train.T, dZ) / m
        db = np.sum(dZ) / m

        # Updating weights and bias (Applying Gradient Descent)
        W -= alpha * dW
        # MUltiplying by 0.1 to decrease bias by time. It improves the weight values otherwise they fluctuate wildly. Learned it th hard way. This is pure experimentaion.
        b -= alpha * db * 0.1

        # Output iteration number and cost value after every 10,000 iterations
        if i % 10_000 == 0:
            print(f"{i} epochs done.")
            print(f"Cost Function Value = {J:.4f}\n")

    print("Final Weights:", W)
    print("Final Bias:", b)

    # Saving Important Files
    np.save(os.path.join(MODEL_DIR, "W.npy"), W)
    np.save(os.path.join(MODEL_DIR, "b.npy"), b)
    np.save(os.path.join(MODEL_DIR, "J.npy"), J)


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_data()
    save_data(X_train, X_test, Y_train, Y_test)
    train(X_train, Y_train)
