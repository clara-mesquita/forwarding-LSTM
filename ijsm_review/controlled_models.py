"""Small, univariate model families for the controlled comparison."""
import math
import numpy as np
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, name, context, config):
        super().__init__()
        self.name = name
        if name == "MLP":
            widths = [context, *config["hidden"], 1]
            layers = []
            for i in range(len(widths) - 1):
                layers.append(nn.Linear(widths[i], widths[i + 1]))
                if i < len(widths) - 2:
                    layers.append(nn.ReLU())
            self.body = nn.Sequential(*layers)
        elif name == "Transformer":
            self.projection = nn.Linear(1, 32)
            positions = torch.arange(context).float().unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, 32, 2) * (-math.log(10000.) / 32))
            encoding = torch.zeros(context, 32)
            encoding[:, 0::2] = torch.sin(positions * frequencies)
            encoding[:, 1::2] = torch.cos(positions * frequencies)
            self.register_buffer("positions", encoding.unsqueeze(0))
            layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64,
                                               dropout=.1, batch_first=True)
            self.body = nn.TransformerEncoder(layer, config["layers"], enable_nested_tensor=False)
            # Independent initializations, as recommended for TransformerEncoder.
            for block in self.body.layers:
                for parameter in block.parameters():
                    if parameter.dim() > 1:
                        nn.init.xavier_uniform_(parameter)
            self.head = nn.Linear(32, 1)
        else:
            recurrent = nn.GRU if name == "GRU" else nn.LSTM
            self.body = recurrent(1, config.get("units", 64),
                                  num_layers=2 if name == "ForwardingLSTM" else 1,
                                  batch_first=True)
            self.head = nn.Linear(config.get("units", 64), 1)

    def forward(self, x):
        if self.name == "MLP":
            return self.body(x).squeeze(-1)
        if self.name == "Transformer":
            encoded = self.body(self.projection(x.unsqueeze(-1)) + self.positions)
            return self.head(encoded[:, -1]).squeeze(-1)
        encoded, _ = self.body(x.unsqueeze(-1))
        return self.head(encoded[:, -1]).squeeze(-1)


def predict(model, window):
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            return float(model(torch.as_tensor(window[None], dtype=torch.float32)).item())
    return float(model.predict(window[None])[0])


def online(model, history, actual):
    """Issue each prediction before reading its target; feedback missing forecasts."""
    state = np.array(history, dtype=float).copy()
    output = []
    for value in actual:
        forecast = state[-1] if model is None else predict(model, state)
        if not np.isfinite(forecast):
            raise ValueError("Non-finite prediction")
        output.append(forecast)
        state = np.concatenate([state[1:], [value if np.isfinite(value) else forecast]])
    return np.asarray(output)
