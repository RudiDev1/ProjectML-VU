import math
import numpy
class NeuralNetwork:
    def __init__(self, activation_function, learning_rate, num_hidden_layers):
        self.activation_function: str = activation_function
        self.learning_rate: float = learning_rate
        self.num_hidden_layers: int = num_hidden_layers
    
    def cost():
        return
    
    def relu(self, x):
        return max(0, x)
    
    def leaky_relu(self, x):
        return max(0.1*x, x)
    
    def sigmoid(self, x):
        return 1 / (1 + (math.e ** -x))

    def feed_forward():
        return
    
    def run_nn():
        return
