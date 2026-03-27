import math
from typing import List
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self, learning_rate, nodes_per_layer, activation_function='RELU'):
        self.activation_function: str = activation_function
        self.learning_rate: float = learning_rate
        self.nodes_per_layer: List[int] = nodes_per_layer
        self.W, _ = self.set_up_wandb()
        _, self.B = self.set_up_wandb()
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vB = [np.zeros_like(b) for b in self.B]
        self.beta = 0.9
    
    def cost(self, model_outputs, expected_outputs, d = 1):
        """
        error = model_outputs - expected_outputs
        mae = (1/np.size(expected_outputs))*np.abs(error)
        return mae
        """
        """
        error = model_outputs - expected_outputsS
        mse = np.mean(np.square(error))
        return mse
        """
        #Huber loss
        error = model_outputs - expected_outputs
        is_small_error  = np.abs(error) <= d
        square_loss = 0.5*np.square(error)
        linear_loss = d*(np.abs(error)-0.5*d)
        return np.mean(np.where(is_small_error, square_loss, linear_loss))
        
    
    def set_up_wandb(self):
        """
        Function to Initialise the Weights and Biases
        Uses the number of nodes per layer to create a array of matrices where each matrix contains the 
        weights for each layer and each bias matrix contains the bias for each layer

        returns an array of matrices of weights and array of matrices of bias
        """
        np.random.seed(1832)
        nodes_per_layer = self.nodes_per_layer
        W = []
        B = []
        for layer in range(1, len(nodes_per_layer)):
            n_in = nodes_per_layer[layer-1]
            n_out = nodes_per_layer[layer]
            w_scale = np.sqrt(2.0/n_in)
            W.append(np.random.randn(n_out, n_in) * w_scale)
            B.append(np.zeros((n_out, 1))) #start with really slow number
        return W, B
    
    def back_prop(self, y_true, d =1):
        n = y_true.shape[1]
        length = len(self.nodes_per_layer) - 1 #total number of layers of weights 1 less than layers of network
        #for mse
        #dA = self.activations[-1] - y_true
        #for huber
        dA = np.where(np.abs(self.activations[-1] - y_true) <= d, self.activations[-1] - y_true, d* np.sign(self.activations[-1] - y_true))
        for i in reversed(range(length)):
            priorA = self.activations[i+1]
            currentA = self.activations[i]
            
            if i == len(self.W) - 1:
                dZ = dA
            else:
                if self.activation_function == 'SIGMOID':
                    dZ = dA * (priorA * (1 - priorA))
                elif self.activation_function == 'RELU':
                    dZ = dA * np.where(priorA > 0, 1, 0.01)
            dW = (1 / n) * (dZ @ currentA.T)
            dB = (1 / n) * np.sum(dZ, axis=1, keepdims=True)

            if i>0:
                dA = self.W[i].T @ dZ
            self.vW[i] = self.beta*self.vW[i] + (1 - self.beta) * dW
            self.vB[i] = self.beta * self.vB[i] + (1 - self.beta) * dB

            self.W[i] -= self.learning_rate * self.vW[i]
            self.B[i] -= self.learning_rate * self.vB[i]
            """
            self.W[i] = self.W[i] - (self.learning_rate * dW)
            self.B[i] = self.B[i] - (self.learning_rate * dB)
            """
    
    def forward(self, model_inputs):
        """
        Runs the forward loop - passes model inputs through network and applies specific weights and biases
        in order to determine the models prediction
        """
        self.activations = [model_inputs]

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

            if layer == len(self.W) - 1:
                activated_output = output
            else:
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
        return np.where(x > 0, x, x * 0.01)
    
    def leaky_relu(self, x):
        return np.maximum(0.1*x, x)
    
    def sigmoid(self, x):
        return 1 / (1 + (math.e ** -x))
    
    def save_model(self):
        np.savez("nn_scratch", *self.W, *self.B)

    def init_model_from_file(self, file):
            data = np.load(file)
            self.W = [data[f'arr_{i}'] for i in range (6)] #due to weight being first 6 arrays
            self.B = [data[f'arr_{i}'] for i in range (6,12)] #due bias being arrays from 6 to 12 
            print("Model loaded")

def prepare_data(path: str) -> dict[str: List]:
    """
    Extract the created dataset from given path and split it into train/test set
    returns Dictionary with string either "train", "test" as key
            List inside at index 0 output (price), index 1 
    """
    df = pd.read_csv(path)
    #select the random seed for a data division
    np.random.seed(1832)
    #handle one-hot embeded data
    df[[c for c in df.columns if c.startswith(('state_id_', 'national_area_'))]] = df[[c for c in df.columns if c.startswith(('state_id_', 'national_area_'))]].values.astype(np.int16)
    #shuffle list of indicies
    indicies = np.arange(len(df))
    np.random.shuffle(indicies)
    #train/test division
    train_size = int(0.8 * len(df))
    val_size = int(0.9*len(df))
    train_i = indicies[:train_size]
    val_i = indicies[train_size:val_size]
    test_i = indicies[val_size:]
    train_set = df.iloc[train_i]
    val_set =df.iloc[val_i]
    test_set = df.iloc[test_i]
    price_train = train_set["price"].values.astype(np.float64)
    price_test = test_set["price"].values.astype(np.float64)
    price_val = val_set["price"].values.astype(np.float64)
    #standardize output
    y_mean = price_train.mean(axis=0)
    y_sd = price_train.std(axis=0)

    y_train = (price_train - y_mean) / y_sd
    y_test = (price_test - y_mean)/ y_sd
    y_val = (price_val - y_mean) / y_sd
    #numerical inputs == cI
    #bed,bath,acre_lot,house_size,state_id,national_area,sectional_center_facility,delivery_area
    cI_train = train_set[["bed", "bath", "acre_lot", "house_size", "sectional_center_facility", "delivery_area"]].values.astype(np.float64)
    cI_test = test_set[["bed", "bath", "acre_lot", "house_size", "sectional_center_facility", "delivery_area"]].values.astype(np.float64)
    cI_val = val_set[["bed", "bath", "acre_lot", "house_size", "sectional_center_facility", "delivery_area"]].values.astype(np.float64)
    #standardize
    mean_tr = cI_train.mean(axis=0)
    sd_tr = cI_train.std(axis=0)
    cI_train = (cI_train - mean_tr) / sd_tr
    cI_test = (cI_test - mean_tr) / sd_tr
    cI_val = (cI_val - mean_tr)/sd_tr
    #"categorical" input == id_I
    train_inputs = np.concatenate([cI_train, train_set[[c for c in df.columns if c.startswith(('state_id_', 'national_area_'))]]], 1)
    test_inputs = np.concatenate([cI_test, test_set[[c for c in df.columns if c.startswith(('state_id_', 'national_area_'))]]], 1)
    val_inputs = np.concatenate([cI_val, val_set[[c for c in df.columns if c.startswith(('state_id_', 'national_area_'))]]], 1)
    return {
        "train": [y_train, train_inputs],
        "val": [y_val, val_inputs],
        "test": [y_test, test_inputs],
        "mean_output": y_mean,
        "sd_output": y_sd
    }

def evaluate(nn, x_test, y_test, mean, sd):
    y_hat_normed = nn.forward(x_test)
    cost = nn.cost(y_hat_normed, y_test)

    y_hat_unnormed = (y_hat_normed * sd) + mean
    y_true_unnormed = (y_test * sd) + mean

    mean_abs_error = np.mean(np.abs(y_hat_unnormed - y_true_unnormed))

    print("--------TEST SET EVALUATION----------")
    print(f"Normalised Huber Loss: {cost}")
    print(f"Mean Absolute Error {mean_abs_error}")
    print(f"Predicted Range: from {np.min(y_hat_unnormed)} to {np.max(y_hat_unnormed)}")
    print(f"Actual Range: from {np.min(y_true_unnormed)} to {np.max(y_true_unnormed)}")
    print(f"Average Prediction: {np.mean(y_hat_unnormed)}")
    print(f"Average Actual Price: {np.mean(y_true_unnormed)}")


def train_validate():
    dict_data = prepare_data("data/clean_estate_data.csv")
    x = dict_data["train"][1]
    y = dict_data["train"][0]
    y = y.reshape(1, -1)
    x = x.T
    nn = NeuralNetwork(0.001, [67, 134, 67, 34, 17, 8, 1])
    epochs = 200
    epoch_arr = []
    losses = []
    
    for i in range(1, epochs+1):
        batches = batching(x, y, 128)
        for x_b, y_b in batches:
            y_hat = nn.forward(x_b)
            cost = nn.cost(y_hat, y_b)
            nn.back_prop(y_b)
        print(f'Epoch: {i} - cost: {cost}')
        epoch_arr.append(i)
        losses.append(cost)
        print(f"""Predict Range: [{(np.min(y_hat)*dict_data["sd_output"]+dict_data["mean_output"]):.4f} to {(np.max(y_hat)*dict_data["sd_output"]+dict_data["mean_output"]):.4f}] 
              | Avg: {(np.mean(y_hat)*dict_data["sd_output"]+dict_data["mean_output"]):.4f}""")

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_arr, losses, label='Training Cost', color='blue')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.show()

    #validation
    x_v = dict_data["val"][1]
    y_v = dict_data["val"][0]
    y_v = y_v.reshape(1, -1)
    x_v = x_v.T
    
    y_hat_v = nn.forward(x_v)
    cost_v = nn.cost(y_hat_v, y_v)
    print(f"""Cost: {cost_v}, 
            #####
            predicted range: [{(np.min(y_hat_v)*dict_data["sd_output"]+dict_data["mean_output"]):.4f} to {(np.max(y_hat_v)*dict_data["sd_output"]+dict_data["mean_output"]):.4f}] 
            sample range: [{(np.min(y_v)*dict_data["sd_output"]+dict_data["mean_output"]):.4f} to {(np.max(y_v)*dict_data["sd_output"]+dict_data["mean_output"]):.4f}] 
            #####
            Avg: {(np.mean(y_hat_v)*dict_data["sd_output"]+dict_data["mean_output"]):.4f}, sample avarage {(np.mean(y_v)*dict_data["sd_output"]+dict_data["mean_output"]):.4f}""")
    nn.save_model()
    return nn
    
    
def evaluate_model():
    check = input("Do you want to run training/validation or model from file? y/n")
    if check == "y":
        nn = train_validate() 
    else:
        nn = get_model()
    dict_data = prepare_data("data/clean_estate_data.csv")
    x_t = dict_data["test"][1]
    y_t = dict_data["test"][0]
    y_t = y_t.reshape(1, -1)
    x_t = x_t.T
    evaluate(nn, x_t, y_t, dict_data["mean_output"], dict_data["sd_output"])
def get_model():
    dict_data = prepare_data("data/clean_estate_data.csv")
    nn =  NeuralNetwork(0.001, [67, 134, 67, 34, 17, 8, 1])
    nn.init_model_from_file("nn_scratch.npz")
    return nn

def batching(x, y, batch_size):
    np.random.seed(1832)
    batches = []
    size = x.shape[1]
    index = np.random.permutation(size)

    new_x = x[:, index]
    new_y = y[:, index]

    for i in range(0, size, batch_size):
        x_batch = new_x[:, i:i+batch_size]
        y_batch = new_y[:, i:i +batch_size]
        batches.append((x_batch, y_batch))
    return batches
evaluate_model()
