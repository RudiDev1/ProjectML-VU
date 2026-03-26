from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import sklearn

from typing import List

import numpy as np
import pandas as pd

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
    #standardize output
    y_mean = price_train.mean(axis=0)
    y_sd = price_train.std(axis=0)

    y_train = (price_train - y_mean) / y_sd
    y_test = (price_test - y_mean)/ y_sd
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
        "train": [y_train, train_inputs],
        "test": [y_train, test_inputs],
        "mean_output": y_mean,
        "sd_output": y_sd
    }

extracted_data = prepare_data("data/clean_estate_data.csv")

model = MLPRegressor(
    hidden_layer_sizes=(134, 67, 34, 17, 8),
    activation='relu',
    solver='sdg',
    learning_rate_init=0.001,
    momentum=0.9,
    batch_size=128,
    max_iter=200,
    random_state=1832,
)

model.fit(extracted_data['train'][1], extracted_data['train'][0])

y_pred = model.predict(extracted_data['test'][1])
y_true = extracted_data['test'][0]

def evaluate(y_true, y_pred):
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    print(f"MSE: {mse} ||| MAE: {mae}")
