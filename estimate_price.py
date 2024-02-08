#!/bin/python3
from ft_linear_regression import linear_regression

def estimate_price(theta0: float, theta1: float):
    mileage = float(input("What mileage ? "))
    estimate_price = linear_regression(theta0, theta1, mileage)
    print(f"Estimate price: {estimate_price}")


if __name__ ==  "__main__":
    theta0 = 0.1418915786374027
    theta1 = 0.9779439287643005
    try:
        estimate_price(theta0, theta1)
    except Exception as e:
        print("./estimate_price.py:", e)