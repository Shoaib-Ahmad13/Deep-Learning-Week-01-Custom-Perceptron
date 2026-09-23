import numpy as np

class Perceptron(object):
    """
    Perceptron classifier utilizing a Sigmoid activation function.
    """
    def __init__(self, eta=0.01, n_iter=10):
        """
        Initializes the Perceptron model.
        
        Parameters:
        eta (float): The learning rate (between 0.0 and 1.0).
        n_iter (int): The maximum number of training iterations over the dataset.
        """
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        """
        Calculates the net input (dot product of inputs and weights, plus bias).
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def sigmoid(self, X):
        """
        Applies the Sigmoid activation function. (Fulfills Task 2)
        """
        return 1.0 / (1.0 + np.exp(-self.weighted_sum(X)))

    def predict(self, X):
        """
        Returns class labels based on a 0.5 threshold applied to the Sigmoid output.
        """
        return np.where(self.sigmoid(X) >= 0.5, 1, 0)

    def fit(self, X, y):
        """
        Fits the training data and updates weights based on prediction errors.
        """
        # Initialize weights to zeros, with one extra element for the bias term
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # Calculate the predicted value
                y_pred = self.predict(xi)
                
                # Compute the update value based on prediction error and learning rate
                update = self.eta * (target - y_pred)
                
                # Update the feature weights
                self.w_[1:] += update * xi
                
                # Update the bias term separately
                self.w_[0] += update
                
                # Increment error count if the prediction was incorrect
                errors += int(update != 0.0)
                
            self.errors_.append(errors)
            
        return self