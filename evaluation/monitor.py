"""
evaluation/monitor.py
Offline model monitoring with EvidentlyAI.

Compares current evaluation results against a baseline run.
Detects verdict distribution drift and accuracy regression.
Generates an HTML report saved to evaluation/monitoring_report.html.

Usage:
    # After running student_eval.py, compare against baseline:
    python evaluation/monitor.py \
        --current  evaluation/results.json \
        --baseline evaluation/results_baseline.json

    # Save baseline from current results (first time setup):
    python evaluation/monitor.py \
        --current  evaluation/results.json \
        --save-baseline
"""

import argparse
import json
import os
import pandas as pd


# ── LOAD RESULTS ─────────────────────────────────────────────────────────────

def load_results(path: str) -> pd.DataFrame:
    """Load evaluation results JSON into a DataFrame."""
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Normalise columns for Evidently
    df["verdict_correct"] = df["verdict_correct"].astype(int)
    df["payout_correct"] = df["payout_correct"].astype(int)
    df["tools_correct"] = df["tools_correct"].astype(int)

    # Encode verdict as binary for classification metrics
    df["verdict_label"] = (df["predicted_verdict"] == "APPROVED").astype(int)
    df["expected_label"] = (df["expected_verdict"] == "APPROVED").astype(int)

    return df


# ── ACCURACY CHECK ────────────────────────────────────────────────────────────

def check_accuracy_regression(current: pd.DataFrame, baseline: pd.DataFrame,
                                threshold: float = 0.05) -> dict:
    """
    Compare verdict accuracy between current and baseline.
    Raises ValueError if regression exceeds threshold.
    """
    current_acc = current["verdict_correct"].mean()
    baseline_acc = baseline["verdict_correct"].mean()
    delta = baseline_acc - current_acc

    result = {
        "baseline_accuracy": round(baseline_acc, 4),
        "current_accuracy":  round(current_acc, 4),
        "delta":             round(delta, 4),
        "threshold":         threshold,
        "regression_detected": delta > threshold,
    }

    print(f"\nAccuracy check:")
    print(f"  Baseline: {baseline_acc:.1%}")
    print(f"  Current:  {current_acc:.1%}")
    print(f"  Delta:    {delta:+.1%}  (threshold: -{threshold:.1%})")

    if result["regression_detected"]:
        raise ValueError(
            f"Verdict accuracy regression detected: "
            f"{baseline_acc:.1%} → {current_acc:.1%} "
            f"(dropped {delta:.1%}, threshold {threshold:.1%})"
        )

    print("  ✓ No regression detected.")
    return result


# ── DISTRIBUTION SUMMARY ──────────────────────────────────────────────────────

def summarise_distributions(current: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """Compare verdict and payout distributions between runs."""

    def dist(df, col):
        return df[col].value_counts(normalize=True).to_dict()

    return {
        "verdict_distribution": {
            "baseline": dist(baseline, "expected_verdict"),
            "current":  dist(current, "predicted_verdict"),
        },
        "accuracy_metrics": {
            "baseline": {
                "verdict_accuracy": round(baseline["verdict_correct"].mean(), 4),
                "payout_precision": round(baseline["payout_correct"].mean(), 4),
                "tool_sequence_acc": round(baseline["tools_correct"].mean(), 4),
            },
            "current": {
                "verdict_accuracy": round(current["verdict_correct"].mean(), 4),
                "payout_precision": round(current["payout_correct"].mean(), 4),
                "tool_sequence_acc": round(current["tools_correct"].mean(), 4),
            },
        },
    }


# ── EVIDENTLY REPORT ──────────────────────────────────────────────────────────

def generate_evidently_report(current: pd.DataFrame, baseline: pd.DataFrame,
                               output_path: str):
    """Generate HTML monitoring report using EvidentlyAI."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import ClassificationPreset, DataDriftPreset
    except ImportError:
        print("EvidentlyAI not installed. Skipping HTML report.")
        print("Install with: pip install evidently")
        return

    # Columns needed for Evidently classification preset
    reference = baseline[["verdict_label", "expected_label"]].rename(
        columns={"verdict_label": "prediction", "expected_label": "target"}
    )
    current_ev = current[["verdict_label", "expected_label"]].rename(
        columns={"verdict_label": "prediction", "expected_label": "target"}
    )

    report = Report(metrics=[
        ClassificationPreset(),
        DataDriftPreset(),
    ])

    report.run(reference_data=reference, current_data=current_ev)
    report.save_html(output_path)
    print(f"\nEvidentlyAI report saved to {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InsureAgent offline monitoring")
    parser.add_argument(
        "--current", default="evaluation/results.json",
        help="Path to current evaluation results JSON"
    )
    parser.add_argument(
        "--baseline", default="evaluation/results_baseline.json",
        help="Path to baseline evaluation results JSON"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Accuracy drop threshold to trigger regression alert (default: 0.05)"
    )
    parser.add_argument(
        "--output", default="evaluation/monitoring_report.html",
        help="Path to save EvidentlyAI HTML report"
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Copy current results to baseline (use after a known-good run)"
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Print summary only, skip EvidentlyAI HTML report"
    )
    args = parser.parse_args()

    # Save baseline mode
    if args.save_baseline:
        import shutil
        shutil.copy(args.current, args.baseline)
        print(f"Baseline saved: {args.current} → {args.baseline}")
        return

    # Load results
    if not os.path.exists(args.current):
        raise FileNotFoundError(f"Current results not found: {args.current}")
    if not os.path.exists(args.baseline):
        raise FileNotFoundError(
            f"Baseline not found: {args.baseline}\n"
            f"Run with --save-baseline first to create one."
        )

    current = load_results(args.current)
    baseline = load_results(args.baseline)

    print(f"Loaded current:  {len(current)} cases from {args.current}")
    print(f"Loaded baseline: {len(baseline)} cases from {args.baseline}")

    # Distribution summary
    summary = summarise_distributions(current, baseline)
    print("\nAccuracy metrics:")
    for run, metrics in summary["accuracy_metrics"].items():
        print(f"  {run}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.1%}")

    # Accuracy regression check — fails loudly if threshold exceeded
    regression = check_accuracy_regression(current, baseline, threshold=args.threshold)

    # Save summary JSON
    summary_path = args.output.replace(".html", "_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({**summary, "regression_check": regression}, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # EvidentlyAI HTML report
    if not args.summary_only:
        generate_evidently_report(current, baseline, args.output)


if __name__ == "__main__":
    main()