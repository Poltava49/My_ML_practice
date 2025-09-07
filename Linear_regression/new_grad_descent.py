import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin


class CustomGradiendDescent(BaseEstimator, RegressorMixin): 
    def __init__(self, loss='mse', learning_rate=0.01, n_iters=1000, penalty=None, alpha=0.0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.penalty = penalty
        self.alpha = alpha
        self.weights = None
        self.bias = None 
        self.loss_history = []

    def fit(self, X, y):
        # Преобразование разреженных матриц в плотные
        if hasattr(X, 'toarray'):
            X = X.toarray()
       
        y = np.array(y).flatten()
        
        # Проверка размерности
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            error = y_pred - y

            loss = np.mean(error**2)
            self.loss_history.append(loss)
            

            dw = (2/n_samples) * np.dot(X.T, error)
            db = (2/n_samples) * np.sum(error)

            # Добавляем регуляризацию к градиентам
            if self.penalty == 'l1':
                dw += self.alpha * np.sign(self.weights)

            self.weights -= self.learning_rate * dw
            self.bias -=self.learning_rate * db
        return self
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def plot_loss(self):
        plt.figure(figsize=(10,6))
        plt.plot(self.loss_history)
        plt.title(f'Градиентный спуск ({self.loss.upper()})')
        plt.xlabel('Итерация')
        plt.ylabel('Значение потерь')
        plt.grid(True)
        plt.show()