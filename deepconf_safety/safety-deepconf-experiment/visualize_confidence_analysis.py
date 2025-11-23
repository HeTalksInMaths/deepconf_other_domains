#!/usr/bin/env python3
"""
Comprehensive confidence distribution analysis for DeepConf safety evaluation.

Creates publication-quality visualizations:
1. Confidence distributions: Correct vs Incorrect predictions (like DeepConf paper)
2. Confidence by refusal category (confident_refusal, uncertain_compliance, etc.)
3. Confidence by ground truth toxicity (toxic vs safe prompts)
4. Trace-level confidence evolution (how confidence changes trace-by-trace)

Usage:
    python visualize_confidence_analysis.py \
        --results-dir results/toxicchat_qwen06b_1000_vllm_reclassified \
        --benchmark toxicchat \
        --output plots/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / 'src'))

from benchmark_loaders import (
    ToxicChatLoader,
    WildGuardMixLoader,
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_predictions(predictions_path: Path) -> List[Dict]:
    """Load predictions from jsonl file."""
    predictions = []
    with predictions_path.open('r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def load_ground_truth(benchmark: str, data_root: Path, split: str) -> Dict[str, str]:
    """Load ground truth labels."""
    if benchmark == 'toxicchat':
        instances = ToxicChatLoader.load(str(data_root / 'toxicchat'), split=split)
    elif benchmark == 'wildguardmix':
        instances = WildGuardMixLoader.load(str(data_root / 'wildguardmix'), split=split)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    return {inst.id: inst.ground_truth_label for inst in instances}


def plot_confidence_by_correctness(predictions: List[Dict], ground_truth: Dict, output_dir: Path):
    """
    Plot confidence distributions for correct vs incorrect predictions.

    Similar to DeepConf paper Figure 3.
    """
    correct_confs = []
    incorrect_confs = []

    for pred in predictions:
        is_correct = pred['final_prediction'] == ground_truth.get(pred['instance_id'])
        # Use all trace confidences, not just average
        confs = pred['confidences']

        if is_correct:
            correct_confs.extend(confs)
        else:
            incorrect_confs.extend(confs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram comparison
    axes[0].hist([correct_confs, incorrect_confs], bins=30, alpha=0.7,
                 label=['Correct', 'Incorrect'], density=True, color=['green', 'red'])
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Confidence Distribution: Correct vs Incorrect Predictions')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # KDE plot
    sns.kdeplot(data=correct_confs, ax=axes[1], label='Correct', color='green', fill=True, alpha=0.3)
    sns.kdeplot(data=incorrect_confs, ax=axes[1], label='Incorrect', color='red', fill=True, alpha=0.3)
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Confidence Distribution (KDE): Correct vs Incorrect')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_by_correctness.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Confidence by Correctness:")
    print(f"  Correct predictions:   mean={np.mean(correct_confs):.3f}, std={np.std(correct_confs):.3f}")
    print(f"  Incorrect predictions: mean={np.mean(incorrect_confs):.3f}, std={np.std(incorrect_confs):.3f}")
    print(f"  Saved: confidence_by_correctness.png")


def plot_confidence_by_category(predictions: List[Dict], output_dir: Path):
    """
    Plot confidence distributions by refusal category.

    Categories: confident_refusal, confident_compliance, uncertain_refusal, uncertain_compliance
    """
    category_confs = {
        'confident_refusal': [],
        'confident_compliance': [],
        'uncertain_refusal': [],
        'uncertain_compliance': []
    }

    for pred in predictions:
        cat = pred.get('confidence_category', 'unknown')
        if cat in category_confs:
            category_confs[cat].extend(pred['confidences'])

    # Filter out empty categories
    category_confs = {k: v for k, v in category_confs.items() if v}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {
        'confident_refusal': 'blue',
        'confident_compliance': 'green',
        'uncertain_refusal': 'orange',
        'uncertain_compliance': 'red'
    }

    for idx, (cat, confs) in enumerate(category_confs.items()):
        axes[idx].hist(confs, bins=30, alpha=0.7, color=colors.get(cat, 'gray'), density=True)
        axes[idx].set_xlabel('Confidence Score')
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'{cat.replace("_", " ").title()}\n(n={len(confs)} traces)')
        axes[idx].axvline(np.mean(confs), color='black', linestyle='--',
                          label=f'Mean: {np.mean(confs):.3f}')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    plt.suptitle('Confidence Distributions by Refusal Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Confidence by Category:")
    for cat, confs in category_confs.items():
        print(f"  {cat:25s}: mean={np.mean(confs):.3f}, std={np.std(confs):.3f}, n={len(confs)}")
    print(f"  Saved: confidence_by_category.png")


def plot_confidence_by_toxicity(predictions: List[Dict], ground_truth: Dict, output_dir: Path):
    """
    Plot confidence distributions by ground truth toxicity.

    Safe prompts vs Toxic prompts
    """
    safe_confs = []
    toxic_confs = []

    for pred in predictions:
        gt_label = ground_truth.get(pred['instance_id'])
        confs = pred['confidences']

        if gt_label == 'safe':
            safe_confs.extend(confs)
        elif gt_label == 'unsafe':
            toxic_confs.extend(confs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram comparison
    axes[0].hist([safe_confs, toxic_confs], bins=30, alpha=0.7,
                 label=['Safe Prompts', 'Toxic Prompts'], density=True,
                 color=['blue', 'red'])
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Confidence Distribution by Ground Truth Toxicity')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # KDE plot
    sns.kdeplot(data=safe_confs, ax=axes[1], label='Safe Prompts', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(data=toxic_confs, ax=axes[1], label='Toxic Prompts', color='red', fill=True, alpha=0.3)
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Confidence Distribution (KDE) by Toxicity')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_by_toxicity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Confidence by Ground Truth Toxicity:")
    print(f"  Safe prompts:  mean={np.mean(safe_confs):.3f}, std={np.std(safe_confs):.3f}, n={len(safe_confs)}")
    print(f"  Toxic prompts: mean={np.mean(toxic_confs):.3f}, std={np.std(toxic_confs):.3f}, n={len(toxic_confs)}")
    print(f"  Saved: confidence_by_toxicity.png")


def plot_trace_evolution(predictions: List[Dict], ground_truth: Dict, output_dir: Path, num_samples: int = 100):
    """
    Plot how confidence evolves trace-by-trace for sample instances.

    Shows whether confidence increases, decreases, or stays stable across traces.
    """
    # Sample instances
    np.random.seed(42)
    sample_indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Separate by correctness and toxicity
    correct_safe = []
    correct_toxic = []
    incorrect_safe = []
    incorrect_toxic = []

    for idx in sample_indices:
        pred = predictions[idx]
        gt = ground_truth.get(pred['instance_id'])
        is_correct = pred['final_prediction'] == gt
        is_toxic = (gt == 'unsafe')

        confs = pred['confidences']

        if is_correct and not is_toxic:
            correct_safe.append(confs)
        elif is_correct and is_toxic:
            correct_toxic.append(confs)
        elif not is_correct and not is_toxic:
            incorrect_safe.append(confs)
        elif not is_correct and is_toxic:
            incorrect_toxic.append(confs)

    # Plot each group
    groups = [
        (correct_safe, 'Correct on Safe Prompts', 'green', axes[0, 0]),
        (correct_toxic, 'Correct on Toxic Prompts', 'blue', axes[0, 1]),
        (incorrect_safe, 'Incorrect on Safe Prompts', 'orange', axes[1, 0]),
        (incorrect_toxic, 'Incorrect on Toxic Prompts', 'red', axes[1, 1])
    ]

    for traces_list, title, color, ax in groups:
        if not traces_list:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Plot individual traces (light)
        for confs in traces_list:
            trace_nums = range(1, len(confs) + 1)
            ax.plot(trace_nums, confs, alpha=0.1, color=color)

        # Plot mean trajectory (bold)
        max_len = max(len(c) for c in traces_list)
        mean_trajectory = []
        for i in range(max_len):
            vals = [c[i] for c in traces_list if i < len(c)]
            if vals:
                mean_trajectory.append(np.mean(vals))

        ax.plot(range(1, len(mean_trajectory) + 1), mean_trajectory,
                color=color, linewidth=3, label='Mean')

        ax.set_xlabel('Trace Number')
        ax.set_ylabel('Confidence')
        ax.set_title(f'{title}\n(n={len(traces_list)} instances)')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

    plt.suptitle('Confidence Evolution Across Traces', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'trace_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Trace Evolution Analysis:")
    print(f"  Analyzed {num_samples} sampled instances")
    print(f"  Saved: trace_evolution.png")


def plot_percentile_comparison(predictions: List[Dict], ground_truth: Dict, output_dir: Path):
    """
    Plot accuracy vs efficiency trade-off at different percentiles.
    """
    # This would require running the percentile sweep
    # For now, we'll load the data from analyze_threshold_sensitivity.py output
    # Or we can create this from the existing analysis
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive confidence distribution analysis"
    )
    parser.add_argument('--results-dir', required=True,
                        help='Path to experiment results directory')
    parser.add_argument('--benchmark', required=True,
                        help='Benchmark name (toxicchat, wildguardmix)')
    parser.add_argument('--data-root', default='data',
                        help='Root folder containing benchmark data')
    parser.add_argument('--split', default='test',
                        help='Dataset split (default: test)')
    parser.add_argument('--output', default='plots',
                        help='Output directory for plots')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples for trace evolution (default: 100)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    predictions_path = results_dir / 'predictions.jsonl'
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        print(f"Error: {predictions_path} not found")
        sys.exit(1)

    print(f"Loading predictions from {predictions_path}...")
    predictions = load_predictions(predictions_path)
    print(f"  Loaded {len(predictions)} predictions")

    print(f"Loading ground truth from {args.benchmark}...")
    ground_truth = load_ground_truth(args.benchmark, Path(args.data_root), args.split)
    print(f"  Loaded {len(ground_truth)} ground truth labels")

    print(f"\nGenerating visualizations...")
    print("=" * 60)

    # Generate all plots
    plot_confidence_by_correctness(predictions, ground_truth, output_dir)
    plot_confidence_by_category(predictions, output_dir)
    plot_confidence_by_toxicity(predictions, ground_truth, output_dir)
    plot_trace_evolution(predictions, ground_truth, output_dir, args.samples)

    print("\n" + "=" * 60)
    print(f"✓ All visualizations saved to {output_dir}/")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
