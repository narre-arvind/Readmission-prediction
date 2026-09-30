"""Create SHAP global and local explanations for the saved best model."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    artifact = joblib.load(ROOT / "models/model_bundle.joblib")
    model = artifact["models"][artifact["best_model"]]
    X = artifact["X_test"].sample(min(100, len(artifact["X_test"])), random_state=42)
    preprocessor = model.named_steps["preprocess"]
    transformed = preprocessor.transform(X)
    names = preprocessor.get_feature_names_out()
    estimator = model.named_steps["model"]
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    background = dense[: min(30, len(dense))]
    explainer = shap.Explainer(lambda values: estimator.predict_proba(values)[:, 1], background, feature_names=names, algorithm="permutation")
    explanation = explainer(dense)
    values = explanation.values
    shap.summary_plot(values, dense, feature_names=names, show=False, max_display=20); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    shap.summary_plot(values, dense, feature_names=names, plot_type="bar", show=False, max_display=20); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()
    importance = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False).head(10)
    importance.to_csv(ROOT / "reports/top_drivers.csv", index=False)
    for index, row in enumerate([0, min(1, len(X) - 1)], start=1):
        shap.plots.waterfall(explanation[row], max_display=12, show=False); plt.tight_layout(); plt.savefig(ROOT / f"reports/figures/patient_{index}_explanation.png", dpi=150, bbox_inches="tight"); plt.close()
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
