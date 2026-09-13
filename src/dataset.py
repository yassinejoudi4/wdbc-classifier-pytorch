# Charge le CSV traité, effectue un split train/val/test stratifié,
# standardise les features et construit les DataLoader PyTorch.
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from pathlib import Path

PROCESSED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "wdbc_processed.csv"


def load_and_split(
    csv_path: Path = PROCESSED_DATA_PATH,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    # Charge le CSV traité et le divise en train/val/test (stratifié) puis standardise les features (fit sur train uniquement).
    
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["diagnosis"]).values
    y = df["diagnosis"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (val_size + test_size),
        stratify=y_temp, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


class WDBCDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


from torch.utils.data import DataLoader

# Construit les DataLoader PyTorch (train/val/test) à partir des splits déjà standardisés.
def get_dataloaders(batch_size):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_and_split()

    train_ds = WDBCDataset(X_train, y_train)
    val_ds = WDBCDataset(X_val, y_val)
    test_ds = WDBCDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler

if __name__ == "__main__":
    train_loader, val_loader, test_loader, scaler = get_dataloaders(batch_size=16)
    X_batch, y_batch = next(iter(train_loader))
    print(X_batch.shape, y_batch.shape)