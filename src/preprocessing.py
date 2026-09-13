import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin"
DATA_URL = f"{BASE_URL}/wdbc.data"
NAMES_URL = f"{BASE_URL}/wdbc.names"

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "wdbc.data"
NAMES_PATH = RAW_DATA_DIR / "wdbc.names"


def _download_file(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Fichier déjà présent : {output_path}")
        return output_path

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    logger.info(f"Sauvegardé : {output_path}")

    return output_path


def download_dataset() -> tuple[Path, Path]:
    data_path = _download_file(DATA_URL, RAW_DATA_PATH)
    names_path = _download_file(NAMES_URL, NAMES_PATH)
    return data_path, names_path





def build_column_names(names_path: Path = NAMES_PATH) -> list[str]:
    """Construit les 32 noms de colonnes à partir de la structure décrite dans wdbc.names."""
    base_features = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
    ]
    feature_columns = (
        [f"{f}_mean" for f in base_features]
        + [f"{f}_se" for f in base_features]
        + [f"{f}_worst" for f in base_features]
    )
    return ["id", "diagnosis"] + feature_columns


def load_raw_data(raw_path: Path = RAW_DATA_PATH, names_path: Path = NAMES_PATH) -> pd.DataFrame:
    column_names = build_column_names(names_path)

    df = pd.read_csv(raw_path, header=None, names=column_names)
    df = df.drop(columns=["id"])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    if df["diagnosis"].isna().any():
        raise ValueError("Valeurs inattendues dans 'diagnosis'.")

    logger.info(f"Shape : {df.shape}")
    logger.info(f"Classes :\n{df['diagnosis'].value_counts()}")

    return df


PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "wdbc_processed.csv"
def save_processed_data(df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset traité sauvegardé : {output_path}")
    return output_path

if __name__ == "__main__":
    download_dataset()
    df = load_raw_data()
    save_processed_data(df)
    print(df.head())