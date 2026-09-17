# Grid search sur hidden_dims et learning rate : entraîne chaque combinaison,
# garde les poids du modèle avec la meilleure val_loss, et sauvegarde uniquement
# la meilleure configuration trouvée sur disque.

import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
from model import WDBCClassifier

MODEL_SAVE_PATH = "models/best_model.pt"

HIDDEN_DIMS_OPTIONS = [[16, 8], [32, 16], [64, 32]]
LEARNING_RATES = [1e-2, 1e-3, 1e-4]


def train_one_config(hidden_dims, lr, epochs, train_loader, val_loader):
    model = WDBCClassifier(hidden_dims=hidden_dims)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    return best_val_loss, best_state


def grid_search(epochs: int = 100, batch_size: int = 16):
    train_loader, val_loader, test_loader, scaler = get_dataloaders(batch_size=batch_size)

    best_overall_loss = float("inf")
    best_config = None
    best_state = None

    for hidden_dims in HIDDEN_DIMS_OPTIONS:
        for lr in LEARNING_RATES:
            val_loss, state = train_one_config(hidden_dims, lr, epochs, train_loader, val_loader)
            print(f"hidden_dims={hidden_dims}, lr={lr} -> val_loss={val_loss:.4f}")

            if val_loss < best_overall_loss:
                best_overall_loss = val_loss
                best_config = (hidden_dims, lr)
                best_state = state

    print(f"\nMeilleure config : hidden_dims={best_config[0]}, lr={best_config[1]}, val_loss={best_overall_loss:.4f}")
    torch.save(best_state, MODEL_SAVE_PATH)

    return best_config


if __name__ == "__main__":
    grid_search()