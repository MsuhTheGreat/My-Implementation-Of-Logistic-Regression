import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Loading breast cancer data from scikit-learn
data = load_breast_cancer()
X, Y = data.data, data.target

# Splitting data into training data and testing data and saving it
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

#Fixed Variables
m, nx = X_train.shape
alpha = 0.00001
epsilon = 1e-8
epochs = 1_000_001

# Initialize weights and bias as normalized
W = np.random.randn(nx, 1)
b = np.random.randn(1, 1)

# Iterate through all m examples
for i in range(epochs):
  # Calculating predicted values
  Z = np.dot(X_train, W) + b
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
  b -= alpha * db * 0.1

  if i % 10_000 == 0:
    print(f"{i} epochs done.")
    print(f"Cost Function Value = {J:.4f}\n")

print("Final Weights:", W)
print("Final Bias:", b)

# Saving Important Files
np.save("W.npy", W)
np.save("b.npy", b)
np.save("J.npy", J)