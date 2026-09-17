# Évalue le modèle sauvegardé sur le test set (jamais vu pendant la grid search) :
# accuracy, precision, recall, F1, ROC-AUC et matrice de confusion.

import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from dataset import get_dataloaders
from model import WDBCClassifier

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pt"


def evaluate(model_path: Path = MODEL_PATH, batch_size: int = 32):  

    train_loader, val_loader, test_loader, scaler = get_dataloaders(batch_size=batch_size)
    checkpoint = torch.load(model_path)
    model = WDBCClassifier(hidden_dims=checkpoint["hidden_dims"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.numpy().flatten())
            all_probs.extend(probs.numpy().flatten())
            all_labels.extend(y_batch.numpy().flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Confusion matrix:\n{cm}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    evaluate()