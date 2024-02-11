import numpy as np
import matplotlib.pyplot as plt

YELLOW = "\033[1;33m"
PURPLE = "\033[1;35m"
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
END = "\033[0m"

class LinearRegression:
    def __init__(self, features, target, iteration=1000, learning_rate=0.1):
        self.theta0 = 0
        self.theta1 = 0
        self.theta0_norm = 0
        self.theta1_norm = 0
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.features = features
        self.target = target
        self.features_norm = self.normalize(features)
        self.target_norm = self.normalize(target)
        self.cost_history = []
        self.gradient_descent()
        self.denormalize_theta(self.features, self.target)

    def display_stat(self):
        print(f"\n{CYAN}Iterations:         {GREEN}{self.iteration}")
        print(f"{CYAN}Learning rate:      {GREEN}{self.learning_rate}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}Theta 0:            {GREEN}{self.theta0}")
        print(f"{CYAN}Theta 1:            {GREEN}{self.theta1}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}Theta 0 normalized: {GREEN}{self.theta0_norm}")
        print(f"{CYAN}Theta 1 normalized: {GREEN}{self.theta1_norm}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}Target: {GREEN}{self.target}")
        print(f"{CYAN}Features: {GREEN}{self.features}{END}")


    @staticmethod
    def predict(theta0, theta1, features):
        return theta0 + (theta1 * features)

    def cost_function(self, theta1, theta0, x_km, y_price):
        m = len(x_km)
        sum_errors = 0
        
        for i in range(0, m):
            y_prediction = self.predict(theta0, theta1, x_km[i])
            sum_errors += (y_prediction - y_price[i])**2
        mse = (1 / (2 * m)) * sum_errors
        return (mse)

    def derivative(self, features, target):
        m = len(features)
        derivative_t0 = 0
        derivative_t1 = 0
        for i in range(0, m):
            target_prediction_norm = self.predict(self.theta0_norm, self.theta1_norm, features[i])
            derivative_t0 += target_prediction_norm - target[i]
            derivative_t1 += (target_prediction_norm - target[i]) * features[i] 
        derivative_t0 /= m
        derivative_t1 /= m
        return derivative_t0, derivative_t1

    def gradient_descent(self):
        for i in range(self.iteration):
            derivative_t0, derivative_t1 = self.derivative(self.features_norm, self.target_norm)
            mse = self.cost_function(self.theta1_norm, self.theta0_norm, self.features_norm, self.target_norm)
            self.cost_history.append(mse)
            if i != 0 and i % (self.iteration / 10) == 0:
                print(f"{CYAN}Iteration:  {YELLOW}{i}{CYAN}| mse: {YELLOW}{mse:.4f} ",
                      f"{CYAN}| theta0: {YELLOW}{self.theta0_norm:.4f} {CYAN}| theta1: {YELLOW}{self.theta1_norm:.4f} {CYAN}|",
                      f"dt0: {YELLOW}{derivative_t0:.4f} {CYAN}| dt1: {YELLOW}{derivative_t1:.4f}{END}")
            self.theta0_norm -= self.learning_rate * (derivative_t0)
            self.theta1_norm -= self.learning_rate * (derivative_t1)

    def normalize(self, array: np.ndarray)-> np.ndarray:
        return (array - np.min(array)) / (np.max(array) - np.min(array))

    def denormalize_theta(self, features, target):
        self.theta1 = self.theta1_norm * (np.max(target) - np.min(target)) / (np.max(features) - np.min(features))
        self.theta0 = np.mean(target) - self.theta1 * np.mean(features)
