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
#         y_prediction = predict(theta0, theta1, x_km[i])
#         sum_errors += (y_prediction - y_price[i])**2
#     mse = (1 / (2 * m)) * sum_errors
#     return (mse)


def predict(theta0, theta1, mileage)-> float:
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
    derivative_t0 = 0
    derivative_t1 = 0
    for i in range(0, m):
        derivative_t0 += predict(t0, t1, x_km[i]) - y_price[i]
        derivative_t1 += (predict(t0, t1, x_km[i]) - y_price[i]) * x_km[i] 
    derivative_t0 /= m
    derivative_t1 /= m
    return derivative_t0, derivative_t1


def gradient_descent(theta0, theta1, x_km, y_price):
    """
    for each iteration:
    Replace old theta by the new theta following this formula:
    w = w - a * (dJ / dw)
    b = b - a * (dJ / db)
    return theta0 and theta1 for a linear regression
    """
    iteration = 1000
    learning_rate = 0.1

    for _ in range(iteration):
        derivative_t0, derivative_t1 = derivative(theta0, theta1, x_km, y_price)
        theta0 = theta0 - learning_rate * (derivative_t0)
        theta1 = theta1 - learning_rate * (derivative_t1)
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


def plot_result(x_km, y_price, x_values, y_values):
    plt.scatter(x_km, y_price, color='blue', label="Training set data")
    plt.plot(x_values, y_values, color='red', label="Regression line")
    plt.grid()
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

    theta0_norm, theta1_norm = gradient_descent(0, 0, x_km_norm, y_price_norm)

    theta1 = theta1_norm * (np.max(y_price) - np.min(y_price)) / (np.max(x_km) - np.min(x_km))
    theta0 = np.mean(y_price) - theta1 * np.mean(x_km)
    
    print(f"theta0 norm: {theta0_norm}, theta1 norm: {theta1_norm}")
    print(f"theta0: {theta0}, theta1: {theta1}")
    x_values = np.linspace(np.min(x_km), np.max(x_km), 100)
    y_values = predict(theta0, theta1, x_values)
    
    plot_result(x_km, y_price, x_values, y_values)

if __name__ ==  "__main__":
    main()