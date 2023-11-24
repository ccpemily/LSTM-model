from torch import optim
from torch import nn
from torch.nn import Module
from tqdm import tqdm

import torch
import numpy as np

from models.lstm import LSTMModule

DEFAULT_MODULE_PARAMS = {
    "max_epoch": 12000,
    "min_loss": 1e-3,
    "loss_func": nn.MSELoss(),
    "optimizer": optim.Adam,
    "lr": 1e-1,
    "batch_size": 4,
    "input_features": 1,
    "output_features": 1,
    "hidden_features": 16,
    "num_layers": 3
}

DEFAULT_DATASET = {
    "train_seq": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32),
    "target_seq": np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=np.float32),
    "predict_seq": np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype=np.float32)
}

def lstm_module(params:dict=DEFAULT_MODULE_PARAMS) -> LSTMModule:
    input_features = params.get("input_features", 1)
    output_features = params.get("output_features", 1)
    hidden_features = params.get("hidden_features", 16)
    num_layers = params.get("num_layers", 3)
    return LSTMModule(input_features, output_features, hidden_features, num_layers)

def train(model:Module, dataset:dict, params:dict=DEFAULT_MODULE_PARAMS) -> None:
    model.train()
    batch_size = params.get("batch_size", 4)
    input_features = params.get("input_features", 1)
    output_features = params.get("output_features", 1)
    input_data = torch.from_numpy(np.reshape(dataset["train_seq"], (-1, batch_size, input_features)))
    output_data = torch.from_numpy(np.reshape(dataset["target_seq"], (-1, batch_size, output_features)))

    loss_func = params.get("loss_func", nn.MSELoss())
    optimizer = params.get("optimizer", optim.Adam)(model.parameters(), lr=params.get("lr", 1e-3))
    for epoch in tqdm(range(params.get('max_epoch', 1000))):
        output = model(input_data)
        loss = loss_func(output, output_data)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch{epoch + 1}/{params.get('max_epoch', 1000)}, Loss: {loss.item()}")

        if loss.item() < params.get("min_loss", 1e-3):
            break

def predict(model:Module, dataset:dict, params:dict=DEFAULT_MODULE_PARAMS) -> np.ndarray:
    model.eval()
    batch_size = params.get("batch_size", 4)
    input_features = params.get("input_features", 1)
    output_features = params.get("output_features", 1)
    input_data = torch.from_numpy(np.reshape(dataset["predict_seq"], (-1, batch_size, input_features)))
    predict_data = model(input_data)
    return predict_data.view(-1, output_features).data.numpy()