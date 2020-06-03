# Harish, Harish
# 2020_03_21


# %tensorflow_version 2.x
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np

class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.inputDimension = input_dimension
        self.layers = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        numOfLayers = len(self.layers)
        if numOfLayers > 0:
            self.layers.append(Perceptron(self.layers[numOfLayers-1].numberOfNodes,num_nodes,transfer_function))
        else:
            self.layers.append(Perceptron(self.inputDimension,num_nodes,transfer_function))

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.layers[layer_number].weights.numpy()

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.layers[layer_number].biases.numpy()

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.layers[layer_number].weights.assign(weights)

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.layers[layer_number].biases.assign(biases)

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.compat.v1.losses.sparse_softmax_cross_entropy(y,y_hat)

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        for layerNumber in range(len(self.layers)):
            X = self.layers[layerNumber].predict(X)
        return X

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size)
        for _ in range(num_epochs):
            for _,(x,y) in enumerate(dataset):
                with tf.GradientTape(persistent=True) as tape:
                    predictions = self.predict(x)
                    loss = self.calculate_loss(y,predictions)
                for layerNumber in range(len(self.layers)):
                    dloss_dw, dloss_db = tape.gradient(loss,  [self.layers[layerNumber].weights,self.layers[layerNumber].biases])
                    self.layers[layerNumber].weights.assign_sub(alpha * dloss_dw)
                    self.layers[layerNumber].biases.assign_sub(alpha * dloss_db)
                del tape

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        predictions = self.predict(X)
        predictedTensor = tf.math.argmax(predictions,axis=1,output_type=tf.dtypes.int32)
        labelTensor = tf.constant(y)
        return 1 - ((tf.math.count_nonzero(tf.math.equal(labelTensor, predictedTensor)).numpy()) / y.shape[0])

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        predictions = self.predict(X)
        predictedTensor = tf.math.argmax(predictions,axis=1,output_type=tf.dtypes.int32)
        labelTensor = tf.constant(y)
        confusionTensor = tf.math.confusion_matrix(labelTensor,predictedTensor,num_classes=10)
        return confusionTensor.numpy()

class Perceptron(object):
    def __init__(self,inputDimensions=2,numberOfNodes=0,transferFunction="linear"):
        self.inputDimensions = inputDimensions
        self.numberOfNodes = numberOfNodes
        self.transferFunction = transferFunction.lower()
        self.weights = tf.Variable(np.random.randn(self.inputDimensions,self.numberOfNodes))
        self.biases = tf.Variable(np.random.randn(1,self.numberOfNodes))
    
    def predict(self,X):
        net = tf.matmul(X,self.weights) + self.biases
        if self.transferFunction == 'linear':
            return net
        elif self.transferFunction == 'relu':
            return tf.nn.relu(net)
        else :
            return tf.math.sigmoid(net)
