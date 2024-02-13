#!/bin/python3

def estimate_price(theta, min_x, max_x, min_y, max_y)-> None:
    mileage = float(input("What mileage ? "))
    if mileage < 0:
        raise ValueError("Mileage can't be negative :(")
    
    mileage_norm = (mileage - min_x) / (max_x - min_x)
    estimate_price_norm = theta[0] + (theta[1] * mileage_norm)
    estimate_price = estimate_price_norm * (max_y - min_y) + min_y
    if estimate_price < 0:
        raise ValueError("Cannot be sell :(")
    print(f"Estimate price: {estimate_price}")

if __name__ ==  "__main__":
    theta = [0, 0]
    min_x1 = 0
    max_x1 = 0
    min_y = 0
    max_y = 0
    try:
        estimate_price(theta, min_x1, max_x1, min_y, max_y)
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        print("./estimate_price.py:", e)