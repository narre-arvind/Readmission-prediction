"""Load and clean the UCI diabetes readmission data."""
from pathlib import Path
import pandas as pd

EXCLUDED_DISCHARGE_CODES = {"11", "13", "14", "19", "20", "21"}
DROP_COLUMNS = {"encounter_id", "patient_nbr"}


def load_data(path: str | Path = "data/diabetic_data.csv") -> pd.DataFrame:
    """Read raw data and represent the dataset's question marks as missing."""
    return pd.read_csv(path, na_values=["?"], low_memory=False)


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply documented leakage controls and return model features and target."""
    frame = data.copy()
    missing_fraction = frame.isna().mean()
    frame = frame.drop(columns=missing_fraction[missing_fraction > 0.40].index)
    frame = frame[~frame["discharge_disposition_id"].astype(str).isin(EXCLUDED_DISCHARGE_CODES)]
    frame = frame.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    target = (frame.pop("readmitted") == "<30").astype(int)
    frame = frame.drop(columns=[column for column in DROP_COLUMNS if column in frame])
    return frame.reset_index(drop=True), target.reset_index(drop=True)


def load_and_clean(path: str | Path = "data/diabetic_data.csv") -> tuple[pd.DataFrame, pd.Series]:
    """Load and clean the raw data in one call."""
    return clean_data(load_data(path))
