# Logistic Regression on Breast Cancer Dataset
A simple logistic regression model to classify breast cancer cases using the Breast Cancer Wisconsin Dataset from `sklearn`.

## 📌 Project Overview
This project trains a logistic regression model using gradient descent and evaluates its performance using:

Practical Accuracy (comparison with actual labels)
My Custom Accuracy Formula (based on cost function sigmoid)
All data and models are saved for reproducibility.

## 📂 Project Structure
```
📦 My-Implementation-Of-Logistic-Regression
│-- 📂 data/               Stores dataset files (X_train.npy, Y_train.npy, etc.)
│-- 📂 models/             Stores trained weights (W.npy, b.npy) and cost function value (J.npy)
|-- 📂 source/             Contain train.py and test.py
    │-- 📜 train.py        Loads data, trains model, and saves parameters
    │-- 📜 test.py         Loads model, makes predictions, evaluates accuracy
│-- 📜 Requirements.txt    Lists dependencies
│-- 📜 README.md           Project documentation (this file)
│-- 📜 .gitignore          Prevents unnecessary files from being committed
```
## ⚡ Installation & Setup
### 1️⃣ Clone the Repository
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```
python -m venv venv
venv\Scripts\activate
```
### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```  
## 🚀 How to Run
### 1️⃣ Train the Model
Run the training script to train the logistic regression model:
```
python train.py
```
Saves the trained model (W.npy, b.npy) inside models/.
Saves the dataset (X_train.npy, Y_train.npy, etc.) inside data/.
### 2️⃣ Test the Model
Run the test script to evaluate model performance:
```
python test.py
```
#### Example Output:
My Measured Accuracy: 91.84%  
Practical Accuracy: 97.81%
## 📊 Accuracy Calculation
This project calculates accuracy in **two ways**:

### ✅ Practical Accuracy
The percentage of correct predictions:

$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \times 100
$$

### 📐 My Custom Made Accuracy Formula
Uses the sigmoid of the cost function:

$$
J_{\text{sigmoid}} = \frac{1}{1 + e^{-J}}
$$

$$
\text{Accuracy} = (1 - J_{\text{sigmoid}}) \times 200
$$

## ⚠️ Limitations & Future Improvements
- *No Feature Scaling*: Standardizing X may improve performance.  
- *Fixed Learning Rate*: Could implement an adaptive learning rate for better convergence.  
- *Limited to Logistic Regression*: Can be extended to more advanced models like Neural Networks.
## 🤝 Contributing
If you’d like to contribute:

1. Fork the repo
2. Create a new branch
3. Commit your changes
4. Open a pull request
## 📜 License
This project is open-source and available under the MIT License.
## 💡 Author
Created by MsuhTheGreat. Feel free to connect! 😊