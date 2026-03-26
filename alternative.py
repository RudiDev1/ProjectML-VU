from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import sklearn

import torch
import torch.nn as nn
import torch.optim as optim

from typing import List

import numpy as np
import pandas as pd
import random


def evaluate(model, extracted_data):
    x_t = torch.tensor(extracted_data['test'][1], dtype=torch.float32)
    y_t_normed = extracted_data['test'][0]

    y_hat_normed = model(x_t).detach().numpy().flatten()

    y_hat_unnorm= y_hat_normed * extracted_data['sd_output'] + extracted_data['mean_output']
    y_true_unnorm = y_t_normed * extracted_data['sd_output'] + extracted_data['mean_output']
    
    mean_sq_error = sklearn.metrics.mean_squared_error(y_true_unnorm, y_hat_unnorm)
    mean_abs_error = sklearn.metrics.mean_absolute_error(y_true_unnorm, y_hat_unnorm)
    print(f"MSE: {mean_sq_error} ||| MAE: {mean_abs_error}")
    print(f"Predicted Range: from {np.min(y_hat_unnorm)} to {np.max(y_hat_unnorm)}")
    print(f"Actual Range: from {np.min(y_true_unnorm)} to {np.max(y_true_unnorm)}")

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
    print(train_inputs)
    return {
        "train": [y_train, train_inputs],
        "val": [y_val, val_inputs],
        "test": [y_test, test_inputs],
        "mean_output": y_mean,
        "sd_output": y_sd
    }

extracted_data = prepare_data("data/clean_estate_data.csv")

random.seed(1832)
np.random.seed(1832)
torch.manual_seed(1832)

"""
//Redundant - doesn't have huber loss available
model = MLPRegressor(
    hidden_layer_sizes=(134, 67, 34, 17, 8),
    activation='relu',
    solver='sgd',
    learning_rate_init=0.001,
    momentum=0.9,
    batch_size=128,
    max_iter=200,
    random_state=1832,
)

model.fit(extracted_data['train'][1], extracted_data['train'][0])

y_pred = model.predict(extracted_data['test'][1])
y_true = extracted_data['test'][0]
"""

model = nn.Sequential(
    nn.Linear(67, 137), nn.LeakyReLU(),
    nn.Linear(137, 67), nn.LeakyReLU(),
    nn.Linear(67, 34), nn.LeakyReLU(),
    nn.Linear(34, 17), nn.LeakyReLU(),
    nn.Linear(17, 8), nn.LeakyReLU(),
    nn.Linear(8, 1),
)

for layer in model:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_in', nonlinearity='relu')
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)

loss = nn.HuberLoss(delta=1.0)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, dampening=0.9)

y_true = extracted_data['test'][0]

x = extracted_data["train"][1]
y = extracted_data["train"][0]
y = y.reshape(1, -1)
x = x.T

epochs = 200
for i in range(epochs+1):
    model.train()
    batches = batching(x, y, 128)
    epoch_loss = 0

    for x_b, y_b in batches:
        optimizer.zero_grad()

        inputs = torch.tensor(x_b.T, dtype=torch.float32)
        targets = torch.tensor(y_b.T, dtype=torch.float32)

        output = model(inputs)
        measured_loss = loss(output, targets)
        measured_loss.backward()
        optimizer.step()

        epoch_loss += measured_loss.item()

    print(f'Epoch: {i} - loss: {epoch_loss/len(batches)}')

evaluate(model, extracted_data)

