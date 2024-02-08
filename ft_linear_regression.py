#!/bin/python3
from load_csv import load
import numpy as np
import matplotlib.pyplot as plt

# Useless function, but good for understanding
# def cost_function(theta1, theta0, x_km, y_price):
#     """
#     J(w, b) = 1 / 2m * sum( f(w,b (x[i]) - y[i] )^2)
#     Return the mean squared error
#     """
#     m = len(x_km)
#     sum_errors = 0
    
#     for i in range(0, m):
#         y_prediction = linear_regression(theta0, theta1, x_km[i])
#         sum_errors += (y_prediction - y_price[i])**2
#     mse = (1 / (2 * m)) * sum_errors
#     return (mse)


def linear_regression(theta0, theta1, mileage)-> float:
    """
    f(w, b, x) = w * x + b
    w and b = theta1 and theta0
    Used to find price prediction with mileage and the parameters theta0 and theta1
    """
    return (theta1 * mileage) + theta0

def derivative(t0, t1, x_km, y_price):
    """
    Get the derivative of theta0 and theta1 follow these formulas
    """
    m = len(x_km)
    tmp_t0 = 0
    tmp_t1 = 0
    for i in range(0, m):
        tmp_t0 += linear_regression(t0, t1, x_km[i]) - y_price[i]
        tmp_t1 += (linear_regression(t0, t1, x_km[i]) - y_price[i]) * x_km[i] 
    derivative_t0 = (1 / m) * tmp_t0
    derivative_t1 = (1 / m) * tmp_t1
    print(f"derivative_t0 {derivative_t0} | derivative_t1 {derivative_t1}")
    return derivative_t0, derivative_t1


def update_theta(t0, t1, x_km, y_price, a):
    """
    Replace old theta by the new theta following this formula:
    w = w - a * (dJ / dw)
    b = b - a * (dJ / db)
    """
    derivative_t0, derivative_t1 = derivative(t0, t1, x_km, y_price)
    new_t0 = t0 - a * (derivative_t0)
    new_t1 = t1 - a * (derivative_t1)
    return new_t0, new_t1


def gradient_descent(theta0, theta1, x_km, y_price):
    """
    for each iteration: update theta
    a = learning rate

    return theta0 and theta1 for a linear regression
    """
    iteration = 10000
    a = 0.0001

    for i in range(iteration):
        new_t0, new_t1 = update_theta(theta0, theta1, x_km, y_price, a)
        theta0 = new_t0
        theta1 = new_t1
    return theta0, theta1


def normalisation(array: np.ndarray)-> np.ndarray:
    """
    Normalize data between 0 and 1 with the following formula:
    Xnorm = (X - Xmin) / (Xmax - Xmin)
    """
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def denormalisation(norm_array: np.ndarray, min, max)-> np.ndarray:
    """
    Denormalize a npdarray normalized
    """
    return norm_array * (max - min) + min

def create_graph(x_km, y_price, x_values, y_values):
    plt.scatter(x_km, y_price, color='blue', label="Training set data")
    plt.plot(x_values, y_values, color='red', label="Regression line")
    plt.xlabel("km")
    plt.ylabel("Price")
    plt.title("Price by km")
    plt.legend()
    plt.show()

def main():
    df = load("data.csv")
    x_km: np.ndarray = df["km"].to_numpy()
    y_price: np.ndarray = df["price"].to_numpy()
    
    x_km_norm = normalisation(x_km)
    y_price_norm = normalisation(y_price)
    theta0, theta1 = gradient_descent(0, 0, x_km, y_price)
    print(f"theta0: {theta0}, theta1: {theta1}")

    # x_values_norm = np.linspace(np.min(x_km_norm), np.max(x_km_norm), 100)
    # y_values_norm = linear_regression(theta0, theta1, x_values_norm)
    # x_values = np.linspace(np.min(x_km), np.max(x_km), 100)
    # y_values = denormalisation(y_values_norm, np.min(y_price), np.max(y_price))
    
    x_values = np.linspace(np.min(x_km), np.max(x_km), 100)
    y_values = linear_regression(theta0, theta1, x_values)
    create_graph(x_km, y_price, x_values, y_values)

if __name__ ==  "__main__":
    main()