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
    X_train_init = model.X_train_norm[:, 1:]
    n = X_train_init.shape[1]
    predictions = model.predict(model.theta_norm, model.X_train_norm)
    
    n_rows = math.ceil((n + 1) / 2)
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, n_rows * 5))
    fig.suptitle("Linear Regression model")
    axs = axs.flatten()
    
    for i in range(n):
        axs[i].scatter(X_train_init[:, i], model.Y_train_norm, color='blue', label="Training set data")
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
    l_curve.set_xlabel("Iterations")
    l_curve.set_ylabel("Cost")
    l_curve.set_title("Learning curve")
    l_curve.grid(True)

    for ax in axs[n+1:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

def main():
    try:
        df = load("data/data.csv")
        # Load X(m x n+1) and Y(m x 1)
        X_train = df.iloc[:, :-1].values
        Y_train = df.iloc[:, -1].values
        headers = list(df.columns)
        Y_train = Y_train.reshape((Y_train.shape[0], 1))
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]

        model = LinearRegression(X_train, Y_train, iteration=1000, learning_rate=0.1)
        # model.display_stat()
        plot_result(model, headers)
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        print(f"./training.py: {e}")

if __name__ ==  "__main__":
    main()