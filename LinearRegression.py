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
        self.features_min = np.min(features)
        self.features_max = np.max(features)
        self.target_min = np.min(target)
        self.target_max = np.max(target)
        self.features_norm = self.normalize(features)
        self.target_norm = self.normalize(target)
        self.gradient_descent()
        self.denormalize_theta(self.features, self.target)

    def display_stat(self):
        print(f"{CYAN}Iterations:         {GREEN}{self.iteration}")
        print(f"{CYAN}Learning rate:      {GREEN}{self.learning_rate}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}Theta 0:            {GREEN}{self.theta0}")
        print(f"{CYAN}Theta 1:            {GREEN}{self.theta1}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}Theta 0 normalized: {GREEN}{self.theta0_norm}")
        print(f"{CYAN}Theta 1 normalized: {self.theta1_norm}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}Target: {GREEN}{self.target}")
        print(f"{CYAN}Features: {GREEN}{self.features}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}Target min:   {GREEN}{self.target_min} | {CYAN}Target max:   {GREEN}{self.target_max}")
        print(f"{CYAN}Features min: {GREEN}{self.features_min} | {CYAN}Features max: {GREEN}{self.features_max}")
        print(f"{END}")

    def predict_norm(self, features):
        return (self.theta1_norm * features) + self.theta0_norm

    def predict(self, features):
        return (self.theta1 * features) + self.theta0
    
    def derivative(self, features, target):
        m = len(features)
        derivative_t0 = 0
        derivative_t1 = 0
        for i in range(0, m):
            derivative_t0 += self.predict_norm(features[i]) - target[i]
            derivative_t1 += (self.predict_norm(features[i]) - target[i]) * features[i] 
        derivative_t0 /= m
        derivative_t1 /= m
        return derivative_t0, derivative_t1
    

    def gradient_descent(self):
        for _ in range(self.iteration):
            derivative_t0, derivative_t1 = self.derivative(self.features_norm, self.target_norm)
            self.theta0_norm -= self.learning_rate * (derivative_t0)
            self.theta1_norm -= self.learning_rate * (derivative_t1)

    @staticmethod
    def normalize(array: np.ndarray)-> np.ndarray:
        return (array - np.min(array)) / (np.max(array) - np.min(array))

    def denormalize_theta(self, features, target):
        self.theta1 = self.theta1_norm * (np.max(target) - np.min(target)) / (np.max(features) - np.min(features))
        self.theta0 = np.mean(target) - self.theta1 * np.mean(features)


    def plot_result(self, title, x_label, y_label):
        x_values = [self.features_min, self.features_max]
        y_values = [self.predict(self.features_min), self.predict(self.features_max)]
        plt.scatter(self.features, self.target, color='blue', label="Training set data")
        plt.plot(x_values, y_values, color='red', label="Regression line")
        plt.grid()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_result_norm(self, title, x_label, y_label):
        x_values = [np.min(self.features_norm), np.max(self.features_norm)]
        y_values = [self.predict_norm(np.min(self.features_norm)), self.predict_norm(np.max(self.features_norm))]
        plt.scatter(self.features_norm, self.target_norm, color='blue', label="Training set data")
        plt.plot(x_values, y_values, color='red', label="Regression line")
        plt.grid()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.show()