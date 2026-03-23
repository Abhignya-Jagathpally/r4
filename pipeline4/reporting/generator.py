"""HTML report generator with embedded figures and tables."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<title>R4-MM Clinical Pipeline Report</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; }}
h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
h2 {{ color: #283593; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th {{ background: #1a237e; color: white; padding: 12px; text-align: left; }}
td {{ padding: 10px; border-bottom: 1px solid #e0e0e0; }}
tr:hover {{ background: #f5f5f5; }}
.metric {{ font-size: 24px; font-weight: bold; color: #1a237e; }}
.card {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
img {{ max-width: 100%; border-radius: 4px; }}
.footer {{ color: #666; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>R4-MM Clinical Outcome Prediction Report</h1>
<p>Generated: {timestamp}</p>
<p>Run ID: {run_id} | Patients: {n_patients} | Features: {n_features}</p>

<h2>Model Comparison</h2>
{model_table}

<h2>Evaluation Details</h2>
{evaluation_details}

<h2>Biomarker Discovery</h2>
{biomarker_section}

<h2>Fairness Audit</h2>
{fairness_section}

<h2>Pipeline Timing</h2>
{timing_table}

<div class="footer">
<p>R4-MM-Clinical Pipeline v0.1.0 | Multi-Modal Clinical Outcome Prediction for Multiple Myeloma</p>
</div>
</body>
</html>"""


class ReportGenerator:
    """Generate HTML report from pipeline results."""

    def __init__(self, config: Any, results_dir: str = "results"):
        self.config = config
        self.results_dir = Path(results_dir)

    def generate(self, context: Dict) -> str:
        """Build full HTML report."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        report = REPORT_TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_id=context.get("run_id", "N/A"),
            n_patients=context.get("n_patients", "N/A"),
            n_features=context.get("n_features", "N/A"),
            model_table=self._model_comparison_table(context.get("evaluation_metrics", {})),
            evaluation_details=self._evaluation_details(context.get("evaluation_metrics", {})),
            biomarker_section=self._biomarker_section(context),
            fairness_section=self._fairness_section(context.get("fairness_results", {})),
            timing_table=self._timing_table(context.get("timings", {})),
        )

        # Generate KM plot
        self._kaplan_meier_plot(context)

        output_path = self.results_dir / "report.html"
        with open(output_path, "w") as f:
            f.write(report)

        logger.info(f"Report written to {output_path}")
        return str(output_path)

    def _model_comparison_table(self, metrics: Dict) -> str:
        rows = ""
        for model, m in metrics.items():
            ci = m.get("c_index", m.get("auroc", "N/A"))
            ci_str = f"{ci:.4f}" if isinstance(ci, float) else str(ci)
            ci_range = ""
            if "c_index_ci" in m:
                ci_range = f" [{m['c_index_ci']['ci_lower']:.4f}, {m['c_index_ci']['ci_upper']:.4f}]"
            rows += f"<tr><td>{model}</td><td class='metric'>{ci_str}{ci_range}</td></tr>\n"
        return f"<table><tr><th>Model</th><th>Primary Metric (95% CI)</th></tr>{rows}</table>"

    def _evaluation_details(self, metrics: Dict) -> str:
        sections = ""
        for model, m in metrics.items():
            sections += f"<div class='card'><h3>{model}</h3><ul>"
            for k, v in m.items():
                if isinstance(v, float):
                    sections += f"<li>{k}: {v:.4f}</li>"
                elif isinstance(v, dict):
                    sections += f"<li>{k}: {v}</li>"
            sections += "</ul></div>"
        return sections

    def _biomarker_section(self, context: Dict) -> str:
        biomarkers = context.get("consensus_biomarkers")
        if biomarkers is None or (isinstance(biomarkers, pd.DataFrame) and biomarkers.empty):
            return "<p>No biomarker results available.</p>"
        return f"<p>Top {len(biomarkers)} consensus biomarkers identified.</p>" + biomarkers.head(20).to_html()

    def _fairness_section(self, fairness: Dict) -> str:
        if not fairness:
            return "<p>No fairness audit results.</p>"
        rows = ""
        for key, val in fairness.items():
            rows += f"<div class='card'><h3>{key}</h3><pre>{val}</pre></div>"
        return rows

    def _timing_table(self, timings: Dict) -> str:
        rows = ""
        for stage, elapsed in timings.items():
            rows += f"<tr><td>{stage}</td><td>{elapsed:.1f}s</td></tr>\n"
        return f"<table><tr><th>Stage</th><th>Duration</th></tr>{rows}</table>"

    def _kaplan_meier_plot(self, context: Dict) -> None:
        """Generate Kaplan-Meier curves by predicted risk group."""
        try:
            from lifelines import KaplanMeierFitter
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            clinical = pd.read_parquet(context.get("clinical_path", ""))
            split_info = context.get("split_info", {})
            test_idx = split_info.get("test", [])
            if not test_idx:
                return

            clinical_test = clinical.iloc[test_idx]
            T = clinical_test["survival_time"].values
            E = clinical_test["event"].values

            # Use first survival model for risk groups
            trained = context.get("trained_models", {})
            for name, model in trained.items():
                if name in ("cox_ph", "deepsurv", "rsf"):
                    features = pd.read_parquet(context["features_path"])
                    X_test = features.iloc[test_idx].values
                    if name == "cox_ph":
                        risk = model.predict(pd.DataFrame(X_test, columns=features.columns))
                    else:
                        risk = model.predict(X_test)

                    # Split into risk tertiles
                    terciles = np.percentile(risk, [33.3, 66.7])
                    groups = np.digitize(risk, terciles)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    kmf = KaplanMeierFitter()
                    for g, label in enumerate(["Low Risk", "Medium Risk", "High Risk"]):
                        mask = groups == g
                        if mask.sum() > 0:
                            kmf.fit(T[mask], E[mask], label=f"{label} (n={mask.sum()})")
                            kmf.plot_survival_function(ax=ax)

                    ax.set_xlabel("Time (months)")
                    ax.set_ylabel("Survival Probability")
                    ax.set_title(f"Kaplan-Meier by {name} Risk Groups")
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(self.results_dir / "figures" / "kaplan_meier.png", dpi=150)
                    plt.close()
                    logger.info("Saved Kaplan-Meier plot")
                    break
        except Exception as e:
            logger.warning(f"KM plot failed: {e}")
