#!/bin/python3
from LinearRegression import LinearRegression
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load(path: str) -> pd.DataFrame:
    HANDLED_ERRORS = (FileNotFoundError, PermissionError,
                      ValueError, IsADirectoryError)
    try:
        df = pd.read_csv(path)
        print(f"Loading dataset of dimensions {df.shape}")
        return df
    except HANDLED_ERRORS as error:
        print(f"{__name__}: {type(error).__name__}: {error}")
        return exit(1)

def plot_result(model, headers):
    X_train_init = model.X_train[:, 1:]
    n = X_train_init.shape[1]
    predictions = model.predict(model.theta, model.X_train)
    
    n_rows = math.ceil((n + 1) / 2)
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, n_rows * 5))
    fig.suptitle("Linear Regression model")
    axs = axs.flatten()
    
    for i in range(n):
        axs[i].scatter(X_train_init[:, i], model.Y_train, color='blue', label="Training set data")
        if i == 0:
            axs[i].plot(X_train_init[:, i], predictions, 'r', label="Predictions")
        else:
            axs[i].plot(X_train_init[:, i], predictions, 'r.', label="Predictions")
        axs[i].set_xlabel(headers[i])
        axs[i].set_ylabel(headers[-1])
        axs[i].set_title(f"{headers[-1]} by {headers[i]}")
        axs[i].legend()
        axs[i].grid()
    
    l_curve = axs[n]
    l_curve.plot(range(len(model.cost_history)), model.cost_history, color='blue')
    l_curve.ylimit = (0, max(model.cost_history))
    l_curve.set_xlabel("Iterations")
    l_curve.set_ylabel("Cost")
    l_curve.set_title("Learning curve")
    l_curve.grid(True)

    for ax in axs[n+1:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

def display_min_max(X_train: np.ndarray, Y_train: np.ndarray):
    for i in range(X_train.shape[1]):
        min_x_feature = np.min(X_train[:, i])
        max_x_feature = np.max(X_train[:, i])
        print(f"min_x{i+1} = {min_x_feature}")
        print(f"max_x{i+1} = {max_x_feature}")
    print(f"min_y = {np.min(Y_train)}")
    print(f"max_y = {np.max(Y_train)}") 

def main():
    try:
        df = load("data/houses.csv")
        X_train = df.iloc[:, :-1].values
        Y_train = df.iloc[:, -1].values
        headers = list(df.columns)
        Y_train = Y_train.reshape((Y_train.shape[0], 1))
        X_train_norm = LinearRegression.normalize(X_train)
        Y_train_norm = LinearRegression.normalize(Y_train)
        X_train_norm = np.c_[np.ones(X_train_norm.shape[0]), X_train_norm]
        model = LinearRegression(X_train_norm, Y_train_norm, iteration=10000, learning_rate=0.1)
        model.display_stat()
        display_min_max(X_train, Y_train)
        plot_result(model, headers)
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        print(f"./training.py: {e}")

if __name__ ==  "__main__":
    main()