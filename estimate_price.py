def estimate_price():
    """
    This program use the following hypothesis to predict the price:

    estimatePrice(mileage) = theta0 + (theta1 * mileage)
    """
    theta_1 = 0
    theta_0 = 0

    mileage = float(input("What mileage ? "))
    estimate_price = theta_0 + (theta_1 * mileage)

    print(estimate_price)

if __name__ ==  "__main__":
    estimate_price()