"""Generate consistent exploratory readmission-rate charts."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from features import add_features
from preprocess import load_data, EXCLUDED_DISCHARGE_CODES

AGE_ORDER = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90+"]
MEDICATION_ORDER = ["1-5", "6-10", "11-15", "16-20", "21-30", "31+"]


def load_eda_data(path: str | Path) -> pd.DataFrame:
    """Apply patient-level and discharge filters while retaining EDA fields."""
    frame = load_data(path)
    frame = frame[~frame["discharge_disposition_id"].astype(str).isin(EXCLUDED_DISCHARGE_CODES)]
    frame = frame.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    frame["readmitted_30d"] = (frame.pop("readmitted") == "<30").astype(int)
    return add_features(frame)


def _medication_bin(value: object) -> str:
    value = float(value)
    if value <= 5:
        return "1-5"
    if value <= 10:
        return "6-10"
    if value <= 15:
        return "11-15"
    if value <= 20:
        return "16-20"
    if value <= 30:
        return "21-30"
    return "31+"


def plot_rate_chart(data: pd.DataFrame, column: str, filename: str, title: str, figures: Path, order: list[str] | None = None) -> None:
    """Save a rate chart with counts, common scale, and overall-rate reference."""
    summary = data.groupby(column, dropna=False)["readmitted_30d"].agg(rate="mean", count="size").reset_index()
    summary[column] = summary[column].fillna("Missing").astype(str)
    if order is not None:
        summary[column] = pd.Categorical(summary[column], categories=order, ordered=True)
        summary = summary.sort_values(column)
    labels = [f"{value}\n(n={count:,})" for value, count in zip(summary[column].astype(str), summary["count"])]
    overall_rate = data["readmitted_30d"].mean()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(range(len(summary)), summary["rate"], color="#2a9d8f")
    axis.axhline(overall_rate, color="#e76f51", linestyle="--", linewidth=1.5, label=f"Overall rate: {overall_rate:.3f}")
    axis.set_xticks(range(len(summary)), labels, rotation=35, ha="right")
    axis.set_ylim(0, 0.2)
    axis.set_xlabel(column.replace("_", " ").title())
    axis.set_ylabel("30-day readmission rate")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / filename, dpi=150)
    plt.close(figure)


def generate_eda_figures(data: pd.DataFrame, figures: str | Path) -> None:
    """Generate all requested EDA charts."""
    figures = Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x="readmitted_30d", ax=axis)
    axis.set_title("30-day readmission distribution")
    axis.set_xlabel("Readmitted within 30 days")
    axis.set_ylabel("Patient count")
    figure.tight_layout(); figure.savefig(figures / "target_distribution.png", dpi=150); plt.close(figure)
    data = data.copy()
    data["medication_bin"] = data["num_medications"].map(_medication_bin)
    plot_rate_chart(data, "age_bucket", "readmission_by_age.png", "30-day readmission rate by age", figures, AGE_ORDER)
    plot_rate_chart(data, "time_in_hospital", "readmission_by_los.png", "30-day readmission rate by length of stay", figures)
    plot_rate_chart(data, "number_inpatient", "readmission_by_prior_inpatient.png", "30-day readmission rate by prior inpatient visits", figures)
    plot_rate_chart(data, "medication_bin", "readmission_by_medications.png", "30-day readmission rate by medication count", figures, MEDICATION_ORDER)
    plot_rate_chart(data, "A1Cresult", "readmission_by_a1c.png", "30-day readmission rate by A1C result", figures)
    plot_rate_chart(data, "change", "readmission_by_medication_change.png", "30-day readmission rate by medication change", figures)
