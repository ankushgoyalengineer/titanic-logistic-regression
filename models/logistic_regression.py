import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionFromScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros((n, 1))
        self.b = 0

        for _ in range(self.epochs):
            z = np.dot(X, self.W) + self.b
            y_pred = self.sigmoid(z)
            loss = -1/m * np.sum(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
            self.losses.append(loss)

            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            self.W -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()
