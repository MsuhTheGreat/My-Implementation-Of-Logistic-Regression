import numpy as np

# Loading necessary variables
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")
W = np.load("W.npy")
b = np.load("b.npy")
J = np.load("J.npy")

# Doing Prediction
Z = np.dot(X_test, W) + b
# Avoid Overflowing
Z = np.clip(Z, -500, 500)
A = 1 / (1 + np.exp(-Z))

# My Formula Accuracy
J_sigmoid = 1 / (1 + np.exp(-J))
accuracy = (1 - J_sigmoid) * 200
print(f"My Measured Accuracy: {accuracy:.2f}%")

#Practical Accuracy
predictions = (A >= 0.5).astype(int)
accuracy = np.mean(predictions == Y_test) * 100
print(f"Practical Accuracy: {accuracy:.2f}%")