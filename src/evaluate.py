"""Evaluate models and produce cost-sensitive business reports."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve)

ROOT = Path(__file__).resolve().parents[1]
MISS_COSTS = (5, 10, 20)


def cost_at_threshold(y_true: pd.Series, probability: pd.Series, threshold: float, miss_cost: int) -> int:
    """Return false-negative cost plus false-positive cost at a threshold."""
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    return int(miss_cost * matrix[1, 0] + matrix[0, 1])


def choose_threshold(y_true: pd.Series, probability: pd.Series, miss_cost: int) -> tuple[float, int]:
    """Choose the lowest-cost observed probability threshold."""
    thresholds = pd.Series(probability).sort_values().unique()
    costs = [(float(threshold), cost_at_threshold(y_true, probability, threshold, miss_cost)) for threshold in thresholds]
    return min(costs, key=lambda item: (item[1], item[0]))


def main() -> None:
    source = ROOT / "models/evaluation_bundle.joblib"
    artifact = joblib.load(source if source.exists() else ROOT / "models/model_bundle.joblib")
    X_test, y_test = artifact["X_test"], artifact["y_test"]
    rows, scores = [], {}
    for name, model in artifact["models"].items():
        scores[name] = model.predict_proba(X_test)[:, 1]
    best_name = artifact["best_model"]
    best_probability = scores[best_name]
    selected_threshold, selected_cost = choose_threshold(y_test, best_probability, miss_cost=5)
    for name, probability in scores.items():
        prediction = (probability >= selected_threshold).astype(int)
        rows.append({"model": name, "roc_auc": roc_auc_score(y_test, probability), "pr_auc": average_precision_score(y_test, probability), "accuracy": accuracy_score(y_test, prediction), "balanced_accuracy": balanced_accuracy_score(y_test, prediction), "precision": precision_score(y_test, prediction, zero_division=0), "recall": recall_score(y_test, prediction, zero_division=0), "f1": f1_score(y_test, prediction, zero_division=0)})
    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results.to_csv(ROOT / "reports/model_comparison.csv", index=False)
    results.set_index("model").plot(kind="bar", figsize=(12, 6), ylim=(0, 1), title=f"Model comparison at threshold {selected_threshold:.4f}")
    plt.tight_layout(); plt.savefig(ROOT / "reports/figures/model_comparison.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 6))
    for name, probability in scores.items():
        fpr, tpr, _ = roc_curve(y_test, probability); plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, probability):.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves"); plt.legend(); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/roc_curves.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 6))
    for name, probability in scores.items():
        precision, recall, _ = precision_recall_curve(y_test, probability); plt.plot(recall, precision, label=f"{name} ({average_precision_score(y_test, probability):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall curves"); plt.legend(); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/pr_curves.png", dpi=150); plt.close()
    best_prediction = (best_probability >= selected_threshold).astype(int)
    plt.figure(figsize=(5, 4)); sns.heatmap(confusion_matrix(y_test, best_prediction, labels=[0, 1]), annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["No readmission", "Readmission"], yticklabels=["No readmission", "Readmission"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix: {best_name}"); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/confusion_matrix.png", dpi=150); plt.close()
    thresholds = pd.Series(best_probability).sort_values().unique()
    threshold_costs = pd.DataFrame([{"threshold": float(threshold), "cost": cost_at_threshold(y_test, best_probability, threshold, 5)} for threshold in thresholds[::max(1, len(thresholds) // 100)]])
    threshold_costs.to_csv(ROOT / "reports/threshold_costs.csv", index=False)
    plt.figure(figsize=(8, 5)); plt.plot(threshold_costs["threshold"], threshold_costs["cost"]); plt.axvline(selected_threshold, color="red", linestyle="--"); plt.xlabel("Decision threshold"); plt.ylabel("Cost (miss=5, false alarm=1)"); plt.title("Cost-sensitive threshold selection"); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/cost_vs_threshold.png", dpi=150); plt.close()
    positive_count = int(y_test.sum()); negative_count = int(len(y_test) - positive_count)
    flag_nobody_cost = 5 * positive_count
    flag_everybody_cost = negative_count
    pd.DataFrame([{"scenario": "chosen_threshold", "threshold": selected_threshold, "miss_cost": 5, "total_cost": selected_cost, "savings_vs_flag_nobody": flag_nobody_cost - selected_cost}, {"scenario": "flag_nobody", "threshold": 1.0, "miss_cost": 5, "total_cost": flag_nobody_cost, "savings_vs_flag_nobody": 0}, {"scenario": "flag_everybody", "threshold": 0.0, "miss_cost": 5, "total_cost": flag_everybody_cost, "savings_vs_flag_nobody": flag_nobody_cost - flag_everybody_cost}]).to_csv(ROOT / "reports/cost_summary.csv", index=False)
    sensitivity_rows = []
    for miss_cost in MISS_COSTS:
        threshold, cost = choose_threshold(y_test, best_probability, miss_cost)
        sensitivity_rows.append({"miss_to_false_alarm_ratio": miss_cost, "best_threshold": threshold, "total_cost": cost, "flag_nobody_cost": miss_cost * positive_count, "savings_vs_flag_nobody": miss_cost * positive_count - cost})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(ROOT / "reports/cost_sensitivity.csv", index=False)
    plt.figure(figsize=(8, 5)); plt.plot(sensitivity["miss_to_false_alarm_ratio"], sensitivity["savings_vs_flag_nobody"], marker="o"); plt.xlabel("Miss-to-false-alarm cost ratio"); plt.ylabel("Savings vs flag nobody"); plt.title("Sensitivity of cost savings to miss cost"); plt.xticks(list(MISS_COSTS)); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/cost_sensitivity.png", dpi=150); plt.close()
    risk_rows = []
    prevalence = y_test.mean()
    for percentage in (10, 20):
        count = max(1, int(len(y_test) * percentage / 100))
        selected = pd.Series(best_probability, index=y_test.index).nlargest(count).index
        group_y = y_test.loc[selected]
        group_precision = float(group_y.mean())
        risk_rows.append({"risk_group": f"top_{percentage}_percent", "patients": count, "readmissions": int(group_y.sum()), "precision": group_precision, "recall": float(group_y.sum() / y_test.sum()), "lift": float(group_precision / prevalence)})
    pd.DataFrame(risk_rows).to_csv(ROOT / "reports/top_risk_groups.csv", index=False)
    deployable = {"models": {best_name: artifact["models"][best_name]}, "X_test": X_test, "y_test": y_test, "feature_schema": artifact["feature_schema"], "best_model": best_name, "best_params": artifact.get("best_params", {}), "decision_threshold": selected_threshold}
    joblib.dump(deployable, ROOT / "models/model_bundle.joblib", compress=3)
    print(results.to_string(index=False)); print(f"Selected threshold: {selected_threshold:.6f}; chosen cost: {selected_cost}; flag nobody: {flag_nobody_cost}; flag everybody: {flag_everybody_cost}")


if __name__ == "__main__":
    main()
