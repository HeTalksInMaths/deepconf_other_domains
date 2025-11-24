#!/usr/bin/env python3
"""
Parse experiment results and extract metrics for conditional logic.

This script reads analysis.json from completed experiments and outputs
decision flags for the orchestrator to use in conditional experiment planning.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def load_analysis(results_dir: str) -> Optional[Dict[str, Any]]:
    """Load analysis.json from results directory."""
    analysis_path = Path(results_dir) / "analysis.json"

    if not analysis_path.exists():
        print(f"Error: {analysis_path} not found", file=sys.stderr)
        return None

    try:
        with open(analysis_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {analysis_path}: {e}", file=sys.stderr)
        return None


def extract_key_metrics(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics for decision making."""
    metrics = {
        "status": "success",
        "overall": {},
        "hypothesis": {},
        "categories": {},
        "decisions": {}
    }

    # Overall metrics
    if "overall" in analysis:
        metrics["overall"] = {
            "accuracy": analysis["overall"].get("accuracy", 0.0),
            "avg_traces": analysis["overall"].get("avg_traces", 0.0),
            "avg_confidence": analysis["overall"].get("avg_confidence", 0.0),
            "token_savings_percent": analysis["overall"].get("token_savings_percent", 0.0),
            "total_tokens": analysis["overall"].get("total_tokens", 0),
            "avg_tokens_per_instance": analysis["overall"].get("avg_tokens_per_instance", 0.0)
        }

    # Hypothesis test results
    if "hypothesis_test" in analysis:
        ht = analysis["hypothesis_test"]
        metrics["hypothesis"] = {
            "uncertain_compliance_accuracy": ht.get("uncertain_compliance_accuracy", 0.0),
            "confident_compliance_accuracy": ht.get("confident_compliance_accuracy", 0.0),
            "difference": ht.get("difference", 0.0),
            "interpretation": ht.get("interpretation", "")
        }

    # Category-specific metrics
    for category in ["uncertain_compliance", "confident_compliance",
                     "uncertain_refusal", "confident_refusal"]:
        if category in analysis:
            cat_data = analysis[category]
            metrics["categories"][category] = {
                "count": cat_data.get("count", 0),
                "accuracy": cat_data.get("accuracy", 0.0),
                "avg_confidence": cat_data.get("avg_confidence", 0.0),
                "avg_traces": cat_data.get("avg_traces", 0.0),
                "total_tokens": cat_data.get("total_tokens", 0)
            }

    return metrics


def make_decisions(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make conditional decisions based on metrics.

    Returns a dictionary of boolean flags for the orchestrator.
    """
    decisions = {
        "hypothesis_supported": False,
        "is_efficient": False,
        "needs_optimization": False,
        "run_wildguardmix": False,
        "run_percentile_sweep": False,
        "run_model_comparison": False,
        "investigate_calibration": False,
        "adjust_max_traces": False,
        "recommended_action": "unknown"
    }

    # Extract key values
    uc_accuracy = metrics["hypothesis"].get("uncertain_compliance_accuracy", 1.0)
    cc_accuracy = metrics["hypothesis"].get("confident_compliance_accuracy", 1.0)
    difference = metrics["hypothesis"].get("difference", 0.0)
    token_savings = metrics["overall"].get("token_savings_percent", 0.0)

    # Decision logic

    # 1. Hypothesis validation
    if uc_accuracy < 0.40 and difference > 0.30:
        decisions["hypothesis_supported"] = True
        decisions["recommended_action"] = "hypothesis_supported"

        # 2. Efficiency check
        if token_savings > 30.0:
            decisions["is_efficient"] = True
            decisions["run_wildguardmix"] = True
            decisions["run_percentile_sweep"] = True
            decisions["run_model_comparison"] = True
            decisions["recommended_action"] = "validate_generalization"
        else:
            decisions["needs_optimization"] = True
            decisions["adjust_max_traces"] = True
            decisions["run_percentile_sweep"] = True
            decisions["recommended_action"] = "optimize_efficiency"

    elif uc_accuracy > 0.70 or difference < 0.10:
        # Hypothesis not supported or weak effect
        decisions["hypothesis_supported"] = False
        decisions["investigate_calibration"] = True
        decisions["run_wildguardmix"] = True  # Check refusal detection
        decisions["recommended_action"] = "investigate_calibration"

    else:
        # Inconclusive results
        decisions["recommended_action"] = "inconclusive_run_more"
        decisions["run_percentile_sweep"] = True

    return decisions


def print_summary(metrics: Dict[str, Any]):
    """Print human-readable summary."""
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)

    print("\n[OVERALL METRICS]")
    overall = metrics["overall"]
    print(f"  Accuracy:           {overall.get('accuracy', 0):.2%}")
    print(f"  Avg Traces:         {overall.get('avg_traces', 0):.2f}")
    print(f"  Token Savings:      {overall.get('token_savings_percent', 0):.1f}%")
    print(f"  Total Tokens:       {overall.get('total_tokens', 0):,}")

    if metrics["hypothesis"]:
        print("\n[HYPOTHESIS TEST]")
        hyp = metrics["hypothesis"]
        print(f"  Uncertain Compliance Accuracy:  {hyp.get('uncertain_compliance_accuracy', 0):.2%}")
        print(f"  Confident Compliance Accuracy:  {hyp.get('confident_compliance_accuracy', 0):.2%}")
        print(f"  Difference:                     {hyp.get('difference', 0):.2%}")
        print(f"  Interpretation: {hyp.get('interpretation', 'N/A')}")

    print("\n[CATEGORY BREAKDOWN]")
    for cat_name, cat_data in metrics["categories"].items():
        print(f"\n  {cat_name.replace('_', ' ').title()}:")
        print(f"    Count:       {cat_data.get('count', 0)}")
        print(f"    Accuracy:    {cat_data.get('accuracy', 0):.2%}")
        print(f"    Avg Traces:  {cat_data.get('avg_traces', 0):.2f}")
        print(f"    Tokens:      {cat_data.get('total_tokens', 0):,}")

    print("\n[DECISIONS]")
    decisions = metrics["decisions"]
    print(f"  Hypothesis Supported:    {decisions.get('hypothesis_supported', False)}")
    print(f"  Is Efficient:            {decisions.get('is_efficient', False)}")
    print(f"  Run WildGuardMix:        {decisions.get('run_wildguardmix', False)}")
    print(f"  Run Percentile Sweep:    {decisions.get('run_percentile_sweep', False)}")
    print(f"  Run Model Comparison:    {decisions.get('run_model_comparison', False)}")
    print(f"\n  → Recommended Action: {decisions.get('recommended_action', 'unknown')}")

    print("\n" + "="*60 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_results.py <results_directory> [--json-only]")
        sys.exit(1)

    results_dir = sys.argv[1]
    json_only = "--json-only" in sys.argv

    # Load analysis
    analysis = load_analysis(results_dir)
    if analysis is None:
        sys.exit(1)

    # Extract metrics
    metrics = extract_key_metrics(analysis)

    # Make decisions
    metrics["decisions"] = make_decisions(metrics)

    # Print summary (unless json-only)
    if not json_only:
        print_summary(metrics)

    # Output JSON for orchestrator
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
