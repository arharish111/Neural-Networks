# Harish, Harish
# 2020-02-29

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.inputDimesions = input_dimensions
        self.numberOfNodes = number_of_nodes
        self.transferFunction = transfer_function.lower()
        self.weights = np.array([])

        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.numberOfNodes,self.inputDimesions)

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if(W.shape != (self.numberOfNodes,self.inputDimesions)):
            return -1
        else:
            self.weights = W

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """

        net = np.dot(self.weights,X)
        if self.transferFunction == 'hard_limit':
            a = (net >= 0).astype(int)
        elif self.transferFunction == 'linear':
            a = net
        return a

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        if X.shape[0] < X.shape[1]:
            X = X.T
            d = np.dot(X.T,X)
            x = np.linalg.inv(d)
            xPseudoInverse = np.dot(x,X.T)
            self.weights = np.dot(y,xPseudoInverse.T)
        else:
            d = np.dot(X.T,X)
            x = np.linalg.inv(d)
            xPseudoInverse = np.dot(x,X.T)
            self.weights = np.dot(y,xPseudoInverse)

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        xPartition = np.array_split(X,batch_size,1)
        yPartition = np.array_split(y,batch_size,1)
        for _ in range(num_epochs):
            for b in range(batch_size):
                a = self.predict(xPartition[b])
                if learning.lower() == 'delta':
                    e = yPartition[b] - a
                    deltaW = alpha * np.dot(e,xPartition[b].T)
                elif learning.lower == 'unsupervised_hebb':
                    deltaW = alpha * np.dot(a,xPartition[b].T)
                else:
                    deltaW = alpha * np.dot(yPartition[b],xPartition[b].T)
                    self.weights = (1-gamma) * self.weights
                self.weights = self.weights + deltaW
            

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        a = self.predict(X)
        diff = y - a
        squaredDiff = diff ** 2
        return np.mean(squaredDiff) 
