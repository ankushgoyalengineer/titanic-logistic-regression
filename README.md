# 🚢 Titanic Survival Prediction (Logistic Regression from Scratch)

This project implements a Logistic Regression classifier from scratch (no ML libraries) to predict passenger survival on the Titanic dataset.

## 📂 Project Structure

titanic_project/
├── data/
│ ├── train.csv
│ └── test.csv
├── models/
│ └── logistic_regression.py
├── utils/
│ └── preprocessing.py
├── main.py
├── submission.csv
├── .env
└── README.md


## 📌 Objective

Predict survival (`Survived`) using basic numerical and categorical features (e.g., age, sex, fare), with manual feature engineering and feature scaling. No use of libraries like `scikit-learn`.

---

## ⚙️ Features

- ✅ Data cleaning and preprocessing
- ✅ Manual feature scaling (standardization)
- ✅ Logistic regression implementation from scratch (NumPy only)
- ✅ Custom training loop with gradient descent
- ✅ Loss tracking and visualization
- ✅ Final prediction on test data and output to CSV

---

## 🧪 How to Run

1. Clone the repo or download the code:
   ```bash
   git clone <your-repo-url>
   cd titanic_project
2. Place train.csv and test.csv inside a data/ folder.
3. Create a .env file: PYTHONPATH=.
4. Run the project: python main.py
