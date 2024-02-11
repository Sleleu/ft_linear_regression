#!/bin/python3
from LinearRegression import LinearRegression

def estimate_price(theta0: float, theta1: float):
    mileage = float(input("What mileage ? "))
    if mileage < 0:
        raise ValueError("Mileage can't be negative :(")
    estimate_price = LinearRegression.predict(theta0, theta1, mileage)
    if estimate_price < 0:
        raise ValueError("Go to the scrapyard")
    print(f"Estimate price: {estimate_price}")


if __name__ ==  "__main__":
    theta0 = 0
    theta1 = 0
    try:
        estimate_price(theta0, theta1)
    except Exception as e:
        print("./estimate_price.py:", e)