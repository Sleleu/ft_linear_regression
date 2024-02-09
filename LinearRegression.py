import numpy as np
import matplotlib.pyplot as plt

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

    def predict_norm(self, features):
        """
        f(w, b, x) = w * x + b
        w and b = theta1 and theta0
        Used to find price prediction with mileage and the parameters theta0 and theta1
        """
        return (self.theta1_norm * features) + self.theta0_norm

    def predict(self, features):
        """
        f(w, b, x) = w * x + b
        w and b = theta1 and theta0
        Used to find price prediction with mileage and the parameters theta0 and theta1
        """
        return (self.theta1 * features) + self.theta0
    
    def derivative(self, features, target):
        """
        Get the derivative of theta0 and theta1 follow these formulas
        """
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
        """
        for each iteration:
        Replace old theta by the new theta following this formula:
        w = w - a * (dJ / dw)
        b = b - a * (dJ / db)
        return theta0 and theta1 for a linear regression
        """
        for _ in range(self.iteration):
            derivative_t0, derivative_t1 = self.derivative(self.features_norm, self.target_norm)
            self.theta0_norm -= self.learning_rate * (derivative_t0)
            self.theta1_norm -= self.learning_rate * (derivative_t1)

    @staticmethod
    def normalize(array: np.ndarray)-> np.ndarray:
        """
        Normalize data between 0 and 1 with the following formula:
        Xnorm = (X - Xmin) / (Xmax - Xmin)
        """
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