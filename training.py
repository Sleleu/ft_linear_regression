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
    x_values = [min(model.features), max(model.features)]
    prediction_target_min = model.predict(model.theta0, model.theta1, min(model.features))
    prediction_target_max = model.predict(model.theta0, model.theta1, max(model.features))
    y_values = [prediction_target_min, prediction_target_max]
    ax1.scatter(model.features, model.target, color='blue', label="Training set data")
    ax1.plot(x_values, y_values, color='red', label="Regression line")
    ax1.grid()
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    ax1.legend()
    
    ax2.plot(range(model.iteration), model.cost_history, color='blue')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Cost")
    ax2.set_title("Cost by iteration")
    ax2.grid()
    plt.show()  

def main():
    try:
        df = load("data.csv")
        x_km = df["km"].to_numpy()
        y_price = df["price"].to_numpy()
        model = LinearRegression(x_km, y_price, iteration=1000, learning_rate=0.1)
        model.display_stat()
        plot_result(model, "Price by km", "Km", "Price")
    except Exception as e:
        print(f"./training.py: {e}")

if __name__ ==  "__main__":
    main()