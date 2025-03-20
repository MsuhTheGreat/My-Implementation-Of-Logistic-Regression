import numpy as np

#Fixed Variables
# m, nx = 100_000, 100
m, nx = 1000, 100
alpha = 0.01
epsilon = 1e-8
epochs = 1_000_001

# Initialize weights and bias as normalized
W = np.random.randn(nx, 1)
b = np.random.randn(1, 1)

# Creating dummy data
X = np.random.randn(nx, m)
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
Y = (np.sum(X, axis=0) > 0).astype(int).reshape(1, m)

# Iterate through all m examples
for i in range(epochs):
  # Calculating predicted values
  Z = np.dot(W.T, X) + b
  A = 1 / (1 + np.exp(-Z))

  # Calculate cost function
  J = -np.mean(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

  # Calculate necessary determinants
  dZ = A - Y
  dW = np.dot(X, dZ.T) / m
  db = np.sum(dZ) / m

  # Updating weights and bias (Applying Gradient Descent)
  W -= alpha * dW
  b -= alpha * db

  if i % 10_000 == 0:
    print(f"{i} epochs done.")
    print(f"Cost Function Value = {J:.4f}\n")

print("Final Weights:", W)
print("Final Bias:", b)

# Saving Important Files
np.save("/content/W.npy", W)
np.save("/content/b.npy", b)
np.save("/content/X.npy", X)
np.save("/content/Y.npy", Y)

#Theoratical Accuracy
J_sigmoid = 1 / (1 + np.exp(-J))
accuracy = (2 * J_sigmoid - 1) * 100
print(f"Theoratical Accuracy: {accuracy:.2f}%")

#Practical Accuracy
predictions = (A >= 0.5).astype(int)
accuracy = np.mean(predictions == Y) * 100
print(f"Practical Accuracy: {accuracy:.2f}%")