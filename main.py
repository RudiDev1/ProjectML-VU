import math
from typing import List
import numpy as np
class NeuralNetwork:
    def __init__(self, learning_rate, num_layers, nodes_per_layer, activation_function='SIGMOID'):
        self.activation_function: str = activation_function
        self.learning_rate: float = learning_rate
        self.num_layers: int = num_layers
        self.nodes_per_layer: List[int] = nodes_per_layer
        self.W, _ = self.set_up_wandb()
        _, self.B = self.set_up_wandb()
    
    def cost():
        return
    
    def set_up_wandb(self):
        """
        Function to Initialise the Weights and Biases
        Uses the number of nodes per layer to create a array of matrices where each matrix contains the 
        weights for each layer and each bias matrix contains the bias for each layer

        returns an array of matrices of weights and array of matrices of bias
        """
        nodes_per_layer = self.nodes_per_layer
        W = []
        B = []
        for layer in range(1, len(nodes_per_layer)):
            W.append(np.random.randn(nodes_per_layer[layer], nodes_per_layer[layer-1]))
            B.append(np.random.randn(nodes_per_layer[layer], 1))
        return W, B
    
    def forward(self, model_inputs):
        """
        Runs the forward loop - passes model inputs through network and applies specific weights and biases
        in order to determine the models prediction
        """
        print(self.W[0])
        print(self.W[0].shape)
        print(model_inputs.shape)
        print(self.B[0].shape)
        output = self.W[0] @ model_inputs + self.B[0]
        if self.activation_function == 'SIGMOID':
            activated_output = self.sigmoid(output)
        elif self.activation_function == 'RELU':
            activated_output = self.relu(output)
        elif self.activation_function == 'LEAKY-RELU':
            activated_output = self.leaky_relu(output)

        for layer in range(len(self.W)-1):
            output = self.W[layer+1] @ activated_output + self.B[layer+1]
            if self.activation_function == 'SIGMOID':
                activated_output = self.sigmoid(output)
            elif self.activation_function == 'RELU':
                activated_output = self.relu(output)
            elif self.activation_function == 'LEAKY-RELU':
                activated_output = self.leaky_relu(output)
        
        y_hat = activated_output
        return y_hat

    
    def relu(self, x):
        return max(0, x)
    
    def leaky_relu(self, x):
        return max(0.1*x, x)
    
    def sigmoid(self, x):
        return 1 / (1 + (math.e ** -x))
    
    def run_nn():
        return

def prepare_training_data():
    return 

def main():
    x = np.array([
        [150, 70],
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69]
    ])
    x = x.T
    nn = NeuralNetwork(0.01, 3, [2, 3, 3, 1])
    y_hat = nn.forward(x)
    print(y_hat)

main()