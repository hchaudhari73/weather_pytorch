import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import pickle
from utils import get_X_y

# class weather_data:
#     def __init__(self, path, target):
#         self.path = path
#         self.target = target
#         self.data = pd.read_csv(self.path)

#     def __getitem__():
#         self.X = self.data.drop(columns=[self.target], axis=1)
#         self.y = self.data[self.target]
#         return self.X, self.y


class TrainModel:

    def __init__(self):
        super().__init__()

    def forward(self, X, w, b):
        self.X = X
        self.w = w
        self.b = b
        self.X = torch.tensor(X.values, requires_grad=True).float()
        self.target = torch.matmul(self.X, self.w) + self.b
        return self.target

    def criteria(self, y_hat, y):
        self.y_hat = y_hat
        # self.y_hat = self.y_hat.detach().numpy()
        self.y = y
        self.y = torch.tensor(y.values, requires_grad=True).float()
        # self.y = self.y.detach().numpy()
        return torch.mean((self.y_hat - self.y) ** 2)

    def fit(self, X_train, y_train, epochs=10, lr=0.01):
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.lr = lr

        # weights and bias
        shape0 = self.X_train.shape[0]
        shape1 = self.X_train.shape[1]
        self.w = 5. * torch.ones(shape1, 1)
        self.w = self.w.clone().detach().requires_grad_(True)
        self.b = torch.tensor(5.0)
        self.b = self.b.clone().detach().requires_grad_(True)
        for e in range(epochs):
            self.y_hat = self.forward(self.X_train, self.w, self.b)
            self.loss = self.criteria(self.y_hat, self.y_train)
            self.loss.backward()

            self.w = self.w.data - lr * self.w.grad.data
            self.b = self.b.data - lr * self.b.grad.data

        return self.y_hat


if __name__ == "__main__":
    from os.path import dirname, abspath, join
    BASE_DIR = dirname(abspath("__file__"))
    DATA_DIR = join(BASE_DIR, "data")
    DATA_PATH = join(DATA_DIR, "clean_data.csv")

    df = pd.read_csv(DATA_PATH)
    df = pd.get_dummies(df)
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42)

    model = TrainModel()
    y_pred = model.fit(X_train, y_train)
    print(classification_report(y_test, y_pred))
