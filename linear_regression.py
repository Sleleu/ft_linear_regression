#!/bin/python3
from load_csv import load
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = load("data.csv")
    x_km = df["km"].to_numpy()
    y_price = df["price"].to_numpy()

    plt.scatter(x_km, y_price, color='blue', label="Training set data")
    plt.xlabel("km")
    plt.ylabel("Price")
    plt.title("Price by km")
    plt.legend()
    plt.show()

if __name__ ==  "__main__":
    main()