#!/bin/python3
from load_csv import load
from math import pow
import numpy as np
import matplotlib.pyplot as plt

def linear_regression(theta0, theta1, mileage)-> float:
    """
    f(w, b, x) = w * x + b
    w and b = theta1 and theta0
    Used to find price prediction with mileage and the parameters theta0 and theta1
    """
    return (theta1 * mileage) + theta0

def cost_function(theta1, theta0, x_km, y_price):
    """
    J(w, b) = 1 / 2m * sum( f(w,b (x[i]) - y[i] )^2)
    Return the mean squared error
    """
    m = len(x_km)
    sum_errors = 0
    
    for i in range(0, m):
        y_prediction = linear_regression(theta0, theta1, x_km[i])
        sum_errors += pow(y_prediction - y_price[i], 2)
    print("sum_errors", sum_errors, "1/2m", (1 / (2 * m)))
    mse = (1 / (2 * m)) * sum_errors
    return (mse)

def main():
    df = load("test.csv")
    x_km = df["km"].to_numpy()
    y_price = df["price"].to_numpy()
    
    theta0 = 0
    theta1 = 1
    mse = cost_function(theta1, theta0, x_km, y_price)
    print(mse)
    plt.scatter(x_km, y_price, color='blue', label="Training set data")
    plt.xlabel("km")
    plt.ylabel("Price")
    plt.title("Price by km")
    plt.legend()
    plt.show()

if __name__ ==  "__main__":
    main()