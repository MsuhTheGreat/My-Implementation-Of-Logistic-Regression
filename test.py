import numpy as np

X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")
W = np.load("W.npy")
b = np.load("b.npy")
J = np.load("J.npy")

Z = np.dot(X_test, W) + b
Z = np.clip(Z, -500, 500)
A = 1 / (1 + np.exp(-Z))

#Theoratical Accuracy
J_sigmoid = 1 / (1 + np.exp(-J))
accuracy = (1 - J_sigmoid) * 200
gpt_accuracy = (1 / (1 + J)) * 100
print(f"My Accuracy: {accuracy:.2f}%")
print(f"GPT Accuracy: {gpt_accuracy:.2f}%")

#Practical Accuracy
predictions = (A >= 0.5).astype(int)
accuracy = np.mean(predictions == Y_test) * 100
print(f"Practical Accuracy: {accuracy:.2f}%")