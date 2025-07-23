import pandas as pd
import numpy as np
from utils.preprocessing import load_and_preprocess_data
from models.logistic_regression import LogisticRegressionFromScratch, plot_loss
import matplotlib.pyplot as plt

# Load and preprocess
X_train, y_train, X_test, passenger_ids = load_and_preprocess_data()

# Train
model = LogisticRegressionFromScratch(lr=0.01, epochs=1000)
model.fit(X_train, y_train)

# Predict
y_test_pred = model.predict(X_test)

# Save submission
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': y_test_pred.flatten()
})
submission.to_csv('submission.csv', index=False)

# Plot training loss
plot_loss(model.losses)
