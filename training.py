#!/bin/python3
from LinearRegression import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load(path: str) -> pd.DataFrame:
    """
    Load a CSV file from the specified path into a Pandas DataFrame.

    Parameters:
    - path (str): The path to the CSV file to be loaded.

    Returns:
    - The loaded DataFrame if successful, None if an error occurs.
    """
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
    x_values = [min(model.features), max(model.features)]
    prediction_target_min = model.predict(model.theta0, model.theta1, min(model.features))
    prediction_target_max = model.predict(model.theta0, model.theta1, max(model.features))
    y_values = [prediction_target_min, prediction_target_max]
    plt.scatter(model.features, model.target, color='blue', label="Training set data")
    plt.plot(x_values, y_values, color='red', label="Regression line")
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
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