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

def plot_result(model: LinearRegression, headers: list, title: str, x_label: str, y_label: str)-> None:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    fig.suptitle("Linear Regression model")

    # delete column of ones
    X_train_init = model.X_train[:, 1:]

    predictions = model.predict(model.theta, model.X_train)
    ax1.scatter(X_train_init, model.Y_train, color='blue', label="Training set data")
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
        X_train = df.iloc[:, :-1].values
        Y_train = df.iloc[:, -1].values
        headers = list(df.columns)
        Y_train = Y_train.reshape((Y_train.shape[0], 1))
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        print(X_train)
        print(Y_train)

        model = LinearRegression(X_train, Y_train, iteration=1000, learning_rate=0.1)
        # model.display_stat()
        plot_result(model, headers, "Price by km", "Km", "Price")
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        print(f"./training.py: {e}")

if __name__ ==  "__main__":
    main()