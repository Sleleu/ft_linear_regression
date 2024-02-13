#!/bin/python3
from numpy import ndarray

def estimate_price(theta: ndarray)-> None:
    mileage = float(input("What mileage ? "))
    if mileage < 0:
        raise ValueError("Mileage can't be negative :(")
    estimate_price = theta[0] + (theta[1] * mileage)
    if estimate_price < 0:
        raise ValueError("Cannot be sell :(")
    print(f"Estimate price: {estimate_price}")


if __name__ ==  "__main__":
    theta0 = 0
    theta1 = 0
    theta = [theta0, theta1]
    try:
        estimate_price(theta)
    except Exception as e:
        print("./estimate_price.py:", e)