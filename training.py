#!/bin/python3
from LinearRegression import LinearRegression
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

def plot_result(model: LinearRegression, title: str, x_label: str, y_label: str)-> None:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    fig.suptitle("Linear Regression model")

    # delete column of ones
    X_train_init = model.X_train_norm[:, :-1]

    predictions = model.predict(model.theta_norm, model.X_train_norm)
    ax1.scatter(X_train_init, model.Y_train_norm, color='blue', label="Training set data")
    ax1.plot(X_train_init, predictions, color='red', label="Regression line")
    ax1.grid()
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    ax1.legend()
    
    ax2.plot(range(model.iteration), model.cost_history, color='blue')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Cost")
    ax2.set_title("Learning curve")
    ax2.grid()
    plt.show()  

def main():
    try:
        df = load("data.csv")

        # Load X(m x n+1) and Y(m x 1)
        X_train = df["km"].to_numpy()
        Y_train = df["price"].to_numpy()
        Y_train = Y_train.reshape((Y_train.shape[0], 1))
        X_train = np.c_[X_train, np.ones(X_train.shape[0])]
        print(X_train.shape)
        print(Y_train.shape)

        model = LinearRegression(X_train, Y_train, iteration=10000, learning_rate=0.1)
        model.display_stat()
        plot_result(model, "Price by km", "Km", "Price")
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        print(f"./training.py: {e}")

if __name__ ==  "__main__":
    main()