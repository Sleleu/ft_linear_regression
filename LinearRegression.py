import numpy as np

YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
END = "\033[0m"

class LinearRegression:
    def __init__(self, X_train, Y_train, iteration=1000, learning_rate=0.1):
        self.iteration = iteration
        self.learning_rate = learning_rate  
        self.cost_history = []
        self.m = X_train.shape[0]     # number of lines in the dataset
        self.n = X_train.shape[1] - 1 # nb of variables to handle (X1, X2, ...)
        self.theta = np.zeros((self.n + 1, 1)) # matrix of parameters
        self.X_train = X_train
        self.Y_train = Y_train
        self.gradient_descent()

    def display_stat(self):
        print(f"\n{CYAN}Iterations:         {GREEN}{self.iteration}")
        print(f"{CYAN}Learning rate:      {GREEN}{self.learning_rate}")
        print(f"\n{YELLOW}|-------------------------------------------|{END}\n")
        print(f"{CYAN}theta = {GREEN}{self.theta.flatten()}{END}")      


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
        Y_prediction = self.predict(self.theta, X_train)
        gradient = (1 / self.m) * np.dot(X_train.T, (Y_prediction - Y_train))
        return gradient

    def gradient_descent(self):
        """
        Gradient descent algorithm in matrix form, following the formula THETA = THETA - alpha * (dJ/dTHETA)
        """
        for i in range(self.iteration):
            gradient = self.gradient(self.X_train, self.Y_train)
            self.theta -= self.learning_rate * gradient
            cost = self.cost_function(self.theta, self.X_train, self.Y_train)
            self.cost_history.append(cost)
            if i != 0 and i % (self.iteration / 10) == 0:
                print(f"{CYAN}Iteration:  {YELLOW}{i}{CYAN} | cost: {YELLOW}{cost:.4f}",
                      f"{CYAN}| theta: {YELLOW}{self.theta.flatten()}",
                      f"{CYAN}| gradient: {YELLOW}{gradient.flatten()}{END}")
    @staticmethod
    def normalize(array: np.ndarray)-> np.ndarray:
        return (array - np.min(array)) / (np.max(array) - np.min(array))