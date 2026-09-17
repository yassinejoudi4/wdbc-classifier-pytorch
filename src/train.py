# Grid search sur hidden_dims et learning rate : entraîne chaque combinaison,
# garde les poids du modèle avec le meilleur F1-score en validation, et sauvegarde
# uniquement la meilleure configuration trouvée sur disque.

import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import Adam

from dataset import get_dataloaders
from model import WDBCClassifier

MODEL_SAVE_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pt"

HIDDEN_DIMS_OPTIONS = [[16, 8], [32, 16], [64, 32]]
LEARNING_RATES = [1e-2, 1e-3, 1e-4]


# Compte les vrais positifs, faux positifs et faux négatifs pour un batch donné
# (calculé manuellement, sans sklearn, pour rester en tensors PyTorch).
def compute_confusion_counts(outputs, y_batch):
    preds = (torch.sigmoid(outputs) > 0.5).float()
    tp = ((preds == 1) & (y_batch == 1)).sum().item()
    fp = ((preds == 1) & (y_batch == 0)).sum().item()
    fn = ((preds == 0) & (y_batch == 1)).sum().item()
    return tp, fp, fn


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0, precision, recall
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def train_one_config(hidden_dims, lr, epochs, train_loader, val_loader):
    model = WDBCClassifier(hidden_dims=hidden_dims)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_val_loss_at_best_f1 = None
    best_val_recall_at_best_f1 = None
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
        val_tp, val_fp, val_fn = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
                tp, fp, fn = compute_confusion_counts(outputs, y_batch)
                val_tp += tp
                val_fp += fp
                val_fn += fn

        val_loss /= len(val_loader.dataset)
        val_f1, val_precision, val_recall = compute_f1(val_tp, val_fp, val_fn)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss_at_best_f1 = val_loss
            best_val_recall_at_best_f1 = val_recall
            best_state = model.state_dict()

    return best_val_f1, best_val_loss_at_best_f1, best_val_recall_at_best_f1, best_state


def grid_search(epochs: int = 100, batch_size: int = 32):
    train_loader, val_loader, test_loader, scaler = get_dataloaders(batch_size=batch_size)

    best_overall_f1 = -1.0
    best_overall_loss = None
    best_overall_recall = None
    best_config = None
    best_state = None

    for hidden_dims in HIDDEN_DIMS_OPTIONS:
        for lr in LEARNING_RATES:
            val_f1, val_loss, val_recall, state = train_one_config(
                hidden_dims, lr, epochs, train_loader, val_loader
            )
            print(f"hidden_dims={hidden_dims}, lr={lr} -> "
                  f"val_f1={val_f1:.4f}, val_recall={val_recall:.4f}, val_loss={val_loss:.4f}")

            if val_f1 > best_overall_f1:
                best_overall_f1 = val_f1
                best_overall_loss = val_loss
                best_overall_recall = val_recall
                best_config = (hidden_dims, lr)
                best_state = state

    print(f"\nMeilleure config : hidden_dims={best_config[0]}, lr={best_config[1]}, "
          f"val_f1={best_overall_f1:.4f}, val_recall={best_overall_recall:.4f}, val_loss={best_overall_loss:.4f}")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
    "state_dict": best_state,
    "hidden_dims": best_config[0],
    "lr": best_config[1],
    }, MODEL_SAVE_PATH)

    return best_config


if __name__ == "__main__":
    grid_search()