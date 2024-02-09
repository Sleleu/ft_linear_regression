#!/bin/python3
from ft_linear_regression import predict

def estimate_price(theta0: float, theta1: float):
    mileage = float(input("What mileage ? "))
    estimate_price = predict(theta0, theta1, mileage)
    print(f"Estimate price: {estimate_price}")


if __name__ ==  "__main__":
    theta0 = 3.2862601528904634e-14
    theta1 = 0.9999999999999946
    try:
        estimate_price(theta0, theta1)
    except Exception as e:
        print("./estimate_price.py:", e)