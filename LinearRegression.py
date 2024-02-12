import numpy as np
import matplotlib.pyplot as plt

YELLOW = "\033[1;33m"
PURPLE = "\033[1;35m"
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
END = "\033[0m"

class LinearRegression:
    def __init__(self, X_train, Y_train, iteration=1000, learning_rate=0.1):
        # self.theta0 = 0
        # self.theta1 = 0
        # self.theta0_norm = 0
        # self.theta1_norm = 0
        self.iteration = iteration
        self.learning_rate = learning_rate
        # self.features = features
        # self.target = target
        # self.features_norm = self.normalize(features)
        # self.target_norm = self.normalize(target)
        
        self.cost_history = []
        self.m = X_train.shape[0]     # number of lines in the dataset
        self.n = X_train.shape[1] - 1 # nb of variables to handle (X1, X2, ...)
        self.theta = np.zeros((self.n + 1, 1)) # matrix of parameters
        self.theta_norm = np.zeros((self.n + 1, 1)) # matrix of normalized parameters

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_train_norm = np.hstack((self.normalize(X_train[:, :-1]), self.X_train[:, -1:]))
        self.Y_train_norm = self.normalize(Y_train)

        self.gradient_descent()
        self.denormalize_theta(self.X_train, self.Y_train)

    def display_stat(self):
        print(f"\n{CYAN}Iterations:         {GREEN}{self.iteration}")
        print(f"{CYAN}Learning rate:      {GREEN}{self.learning_rate}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}Theta:\n            {GREEN}{self.theta}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}Theta 0 normalized: {GREEN}{self.theta_norm[0]}")
        print(f"{CYAN}Theta 1 normalized: {GREEN}{self.theta_norm[1]}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}Target: {GREEN}{self.Y_train}")
        print(f"{CYAN}Features: {GREEN}{self.X_train}{END}")


    @staticmethod
    def predict(theta: np.ndarray, X_train: np.ndarray)->np.ndarray:
        """
        Return a matrix(n x 1) with all predictions
        following the hypothesis F = X . THETA, matrix form of y = ax + b
        """
        return np.dot(X_train, theta)

    def cost_function(self, theta: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray)-> float:
        """
        Return a scalar, representing the mean cost of predictions - target data,
        following the matrix formula:  J(THETA) = 1 / 2m * sum(X . THETA - Y)2
        """
        Y_prediction = self.predict(theta, X_train)
        errors = (Y_prediction - Y_train)**2
        cost = (1 / (2 * self.m)) * np.sum(errors)
        return (cost)

    def gradient(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Return a matrix(n+1, 1) representing the gradient
        """
        Y_prediction = self.predict(self.theta_norm, X_train)
        gradient = (1 / self.m) * np.dot(X_train.T, (Y_prediction - Y_train))
        return gradient

    def gradient_descent(self):
        for i in range(self.iteration):
            gradient = self.gradient(self.X_train_norm, self.Y_train_norm)
            self.theta_norm -= self.learning_rate * gradient
            cost = self.cost_function(self.theta_norm, self.X_train_norm, self.Y_train_norm)
            self.cost_history.append(cost)
            if i != 0 and i % (self.iteration / 10) == 0:
                print(f"{CYAN}Iteration:  {YELLOW}{i}{CYAN}| cost: {YELLOW}{cost:.4f} ",
                      f"gradient0: {YELLOW}{gradient[0]} {CYAN}{END}")

    def normalize(self, array: np.ndarray)-> np.ndarray:
        return (array - np.min(array)) / (np.max(array) - np.min(array))

    def denormalize_theta(self, X_train, Y_train):
        self.theta[:-1] = self.theta_norm[:-1] * (np.max(Y_train) - np.min(Y_train)) / (np.max(X_train[:,: -1]) - np.min(X_train[:,:-1]))
        self.theta[-1] = np.mean(Y_train) - self.theta[:-1] * np.mean(X_train[:, : -1])