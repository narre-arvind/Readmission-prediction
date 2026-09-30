"""Feature engineering and preprocessing utilities."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AGE_MAP = {"[0-10)": "0-10", "[10-20)": "10-20", "[20-30)": "20-30", "[30-40)": "30-40", "[40-50)": "40-50", "[50-60)": "50-60", "[60-70)": "60-70", "[70-80)": "70-80", "[80-90)": "80-90", "[90-100)": "90+"}


def diagnosis_group(value: object) -> str:
    """Map ICD-9 prefixes into clinically meaningful broad groups."""
    text = str(value)
    if text in {"nan", "None"}:
        return "Missing"
    if text.startswith("250"):
        return "Diabetes"
    if text.startswith(("390", "391", "392", "393", "394", "395", "396", "397", "398", "401", "402", "403", "404", "405", "410", "411", "412", "413", "414", "415", "416", "417", "418", "420", "421", "422", "423", "424", "425", "426", "427", "428", "429", "430", "431", "432", "433", "434", "435", "436", "437", "438", "440", "441", "442", "443", "444", "445", "446", "447", "448", "449")):
        return "Circulatory"
    if text.startswith(("460", "461", "462", "463", "464", "465", "466", "467", "468", "469", "470", "471", "472", "473", "474", "475", "476", "477", "478", "480", "481", "482", "483", "484", "485", "486", "487", "488", "489", "490", "491", "492", "493", "494", "495", "496", "497", "498", "499", "500", "501", "502", "503", "504", "505", "506", "507", "508", "509", "510", "511", "512", "513", "514", "515", "516", "517", "518", "519", "520", "521", "522", "523", "524", "525", "526", "527", "528", "529", "530", "531", "532", "533", "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "544", "545", "546", "547", "548", "549", "550", "551", "552", "553", "554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566", "567", "568", "569", "570", "571", "572", "573", "574", "575", "576", "577", "578", "579", "580", "581", "582", "583", "584", "585", "586", "587", "588", "589", "590", "591", "592", "593", "594", "595", "596", "597", "598", "599")):
        return "Respiratory"
    if text.startswith(("140", "141", "142", "143", "144", "145", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200", "201", "202", "203", "204", "205", "206", "207", "208", "209")):
        return "Neoplasm"
    if text.startswith(("580", "581", "582", "583", "584", "585", "586", "587", "588", "589", "590", "591", "592", "593", "594", "595", "596", "597", "598", "599")):
        return "Genitourinary"
    if re.match(r"^[0-9]", text):
        return "Other"
    return "Unknown"


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create compact clinical features while retaining useful original fields."""
    result = frame.copy()
    if "age" in result:
        result["age_bucket"] = result["age"].map(AGE_MAP).fillna(result["age"])
    diagnosis_columns = [column for column in ["diag_1", "diag_2", "diag_3"] if column in result]
    for column in diagnosis_columns:
        result[f"{column}_group"] = result[column].map(diagnosis_group)
    prior_columns = [column for column in ["number_outpatient", "number_emergency", "number_inpatient"] if column in result]
    result["total_prior_visits"] = result[prior_columns].fillna(0).sum(axis=1) if prior_columns else 0
    medication_columns = [column for column in ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"] if column in result]
    result["medications_changed"] = result[medication_columns].isin(["Up", "Down"]).sum(axis=1) if medication_columns else 0
    result = result.drop(columns=diagnosis_columns, errors="ignore")
    return result


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a train-fittable numeric/categorical preprocessing transformer."""
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), categorical),
    ])
