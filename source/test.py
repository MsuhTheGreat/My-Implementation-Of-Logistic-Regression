import numpy as np
import os

DATA_DIR = "data"
MODEL_DIR = "models"


def load_test_data():
    """Load test dataset and trained parameters."""
    # Loading necessary variables
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))
    W = np.load(os.path.join(MODEL_DIR, "W.npy"))
    b = np.load(os.path.join(MODEL_DIR, "b.npy"))
    J = np.load(os.path.join(MODEL_DIR, "J.npy"))
    return X_test, Y_test, W, b, J


def predict(X_test, W, b):
    """Make predictions using logistic regression."""
    # Doing Prediction
    Z = np.dot(X_test, W) + b
    # Avoid Overflowing
    Z = np.clip(Z, -500, 500)
    A = 1 / (1 + np.exp(-Z))
    return A


def evaluate_model(Y_test, A, J):
    """Evaluate accuracy using custom and practical formulas."""
    # My Custom Accuracy Formula
    J_sigmoid = 1 / (1 + np.exp(-J))
    my_accuracy = (1 - J_sigmoid) * 200
    print(f"My Measured Accuracy: {my_accuracy:.2f}%")

    #Practical Accuracy
    predictions = (A >= 0.5).astype(int)
    accuracy = np.mean(predictions == Y_test) * 100
    print(f"Practical Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    X_test, Y_test, W, b, J = load_test_data()
    A = predict(X_test, W, b)
    evaluate_model(Y_test, A, J)
    