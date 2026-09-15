# Définit l'architecture du MLP pour la classification binaire WDBC.

import torch.nn as nn


class WDBCClassifier(nn.Module):
    def __init__(self, input_dim: int = 30, hidden_dims: list[int] = [64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    import torch

    model = WDBCClassifier()
    dummy_input = torch.randn(4, 30)
    output = model(dummy_input)
    print(output.shape)