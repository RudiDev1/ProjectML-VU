import math
from typing import List
import numpy as np
import pandas as pd
class NeuralNetwork:
    def __init__(self, learning_rate, nodes_per_layer, activation_function='RELU'):
        self.activation_function: str = activation_function
        self.learning_rate: float = learning_rate
        self.nodes_per_layer: List[int] = nodes_per_layer
        self.W, _ = self.set_up_wandb()
        _, self.B = self.set_up_wandb()
    
    def cost(self, model_outputs, expected_outputs):
        losses = np.abs(model_outputs - expected_outputs)
        summed_losses = np.sum(losses)
        num_of_data = model_outputs.reshape(-1).shape[0]
        average_losses = (1 / num_of_data) * summed_losses
        return average_losses
    
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
    
    def back_prop(self, y_true):
        n = y_true.shape[1]
        length = len(self.nodes_per_layer) - 1 #total number of layers of weights 1 less than layers of network

        dA = self.activations[-1] - y_true

        for i in reversed(range(length)):
            priorA = self.activations[i+1]
            currentA = self.activations[i]

            dZ = dA * (priorA * (1 - priorA))
            dW = (1 / n) * (dZ @ currentA.T)
            dB = (1 / n) * np.sum(dZ, axis=1, keepdims=True)

            if i>0:
                dA = self.W[i].T @ dZ

            self.W[i] = self.W[i] - (self.learning_rate * dW)
            self.B[i] = self.B[i] - (self.learning_rate * dB)
    
    def forward(self, model_inputs):
        """
        Runs the forward loop - passes model inputs through network and applies specific weights and biases
        in order to determine the models prediction
        """
        self.activations = [model_inputs]

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
        
        self.activations.append(activated_output)

        for layer in range(len(self.W)-1):
            output = self.W[layer+1] @ activated_output + self.B[layer+1]
            if self.activation_function == 'SIGMOID':
                activated_output = self.sigmoid(output)
            elif self.activation_function == 'RELU':
                activated_output = self.relu(output)
            elif self.activation_function == 'LEAKY-RELU':
                activated_output = self.leaky_relu(output)
            
            self.activations.append(activated_output)
        
        y_hat = activated_output
        return y_hat

    
    def relu(self, x):
        return np.max(0, x)
    
    def leaky_relu(self, x):
        return np.max(0.1*x, x)
    
    def sigmoid(self, x):
        return 1 / (1 + (math.e ** -x))
    
    def run_nn():
        return

def prepare_data(path: str) -> dict[str: List]:
    """
    Extract the created dataset from given path and split it into train/test set
    returns Dictionary with string either "train", "test" as key
            List inside at index 0 output (price), index 1 
    """
    df = pd.read_csv(path)
    #select the random seed for a data division
    np.random.seed(142344)
    #shuffle list of indicies
    indicies = np.arange(len(df))
    np.random.shuffle(indicies)
    #train/test division
    train_size = int(0.8 * len(df))
    train_i = indicies[:train_size]
    test_i = indicies[train_size:]
    train_set = df.iloc[train_i]
    test_set = df.iloc[test_i]
    price_train = train_set["price"].values.astype(np.float64)
    price_test = test_set["price"].values.astype(np.float64)
    #numerical inputs == cI
    #bed,bath,acre_lot,house_size,state_id,national_area,sectional_center_facility,delivery_area
    cI_train = train_set[["bed", "bath", "acre_lot", "house_size"]].values.astype(np.float64)
    cI_test = test_set[["bed", "bath", "acre_lot", "house_size"]].values.astype(np.float64)
    #standardize
    mean_tr = cI_train.mean(axis=0)
    sd_tr = cI_train.std(axis=0)
    cI_train = (cI_train - mean_tr) / sd_tr
    cI_test = (cI_test - mean_tr) / sd_tr
    #"categorical" input == id_I
    id_i_train = train_set[["state_id","national_area","sectional_center_facility","delivery_area"]].values.astype(np.int16)
    id_i_test = test_set[["state_id","national_area","sectional_center_facility","delivery_area"]].values.astype(np.int16)
    train_inputs = np.concatenate([cI_train, id_i_train], 1)
    test_inputs = np.concatenate([cI_test, id_i_test], 1)
    return {
        "train": [price_train, train_inputs],
        "test": [price_test, test_inputs]
    }

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
    x = x / np.max(x)
    y = np.array([[1, 0, 0, 1, 1, 0, 0, 1, 0, 1]])

    x = x.T
    nn = NeuralNetwork(0.01, [2, 3, 3, 1])
    

def main():
    dict_data = prepare_data("data/clean_estate_data.csv")
    x = dict_data["train"][1]
    y = dict_data["train"][0]
    y = y.reshape(1, -1)
    x = x.T
    nn = NeuralNetwork(0.01, [8, 3, 1])

    epochs = 100
    for i in range(epochs):
        y_hat = nn.forward(x)
        cost = nn.cost(y_hat, y)
        nn.back_prop(y)
        print(f'Epoch: {i} - cost: {cost}')
main()
