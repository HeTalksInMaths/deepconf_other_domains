#!/usr/bin/env python3
"""
Create comprehensive safety-focused visualizations

Generates:
1. 2x2 Confusion Matrix (Toxic/Safe × Refused/Answered)
2. Sensitivity/Specificity curves across percentiles
3. All original confidence distribution plots
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from benchmark_loaders import ToxicChatLoader, WildGuardMixLoader

sns.set_style("whitegrid")


def load_data(predictions_path, data_path, benchmark='toxicchat'):
    """Load predictions and ground truth."""
    # Load predictions
    predictions = []
    with predictions_path.open('r') as f:
        for line in f:
            predictions.append(json.loads(line))

    # Load ground truth
    if benchmark == 'toxicchat':
        instances = ToxicChatLoader.load(str(data_path / 'toxicchat'), split='test')
    elif benchmark == 'wildguardmix':
        instances = WildGuardMixLoader.load(str(data_path / 'wildguardmix'), split='test')
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    ground_truth = {inst.id: inst.ground_truth_label for inst in instances}

    return predictions, ground_truth


def plot_confusion_matrix_2x2(predictions, ground_truth, output_dir):
    """
    Plot 2x2 confusion matrix: Ground Truth × Final Prediction

    Quadrants:
    - Top-left: Toxic + Refused (TP - correctly caught)
    - Top-right: Toxic + Answered (FN - DANGEROUS, missed)
    - Bottom-left: Safe + Refused (FP - false alarm)
    - Bottom-right: Safe + Answered (TN - correctly allowed)
    """
    # Count each category
    tp, tn, fp, fn = 0, 0, 0, 0

    for pred in predictions:
        gt = ground_truth.get(pred['instance_id'])
        final_pred = pred['final_prediction']

        is_toxic = (gt == 'unsafe')
        is_flagged = (final_pred == 'unsafe')

        if is_toxic and is_flagged:
            tp += 1  # Toxic + Refused
        elif not is_toxic and not is_flagged:
            tn += 1  # Safe + Answered
        elif not is_toxic and is_flagged:
            fp += 1  # Safe + Refused
        else:  # is_toxic and not is_flagged
            fn += 1  # Toxic + Answered (DANGEROUS)

    # Create confusion matrix
    cm = np.array([[tp, fn],
                   [fp, tn]])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use red for dangerous quadrant (FN)
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)

    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True,
                xticklabels=['Flagged (Unsafe)', 'Allowed (Safe)'],
                yticklabels=['Toxic Prompt', 'Safe Prompt'],
                ax=ax, annot_kws={'size': 20, 'weight': 'bold'})

    ax.set_title('Safety Confusion Matrix: Ground Truth × Model Prediction',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model Prediction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')

    # Add quadrant labels
    ax.text(0.25, 0.25, 'TRUE POSITIVE\n(Correctly Caught)',
            ha='center', va='center', fontsize=11, style='italic', color='white')
    ax.text(1.25, 0.25, 'FALSE NEGATIVE\n⚠️ DANGEROUS ⚠️\n(Missed Toxic)',
            ha='center', va='center', fontsize=11, style='italic', color='white', weight='bold')
    ax.text(0.25, 1.25, 'FALSE POSITIVE\n(False Alarm)',
            ha='center', va='center', fontsize=11, style='italic', color='black')
    ax.text(1.25, 1.25, 'TRUE NEGATIVE\n(Correctly Allowed)',
            ha='center', va='center', fontsize=11, style='italic', color='black')

    # Add metrics
    total = tp + tn + fp + fn
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    metrics_text = (
        f"Accuracy: {accuracy*100:.1f}%\n"
        f"Sensitivity (Catch Rate): {sensitivity*100:.1f}%\n"
        f"Specificity (Allow Rate): {specificity*100:.1f}%\n"
        f"Missed Toxic: {fn} ({fn/(tp+fn)*100:.1f}% of toxic)"
    )

    ax.text(1.02, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=11, va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / 'confusion_matrix_2x2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def plot_percentile_curves(percentile_results, output_dir):
    """
    Plot accuracy, sensitivity, specificity, and savings vs percentile.

    Shows trade-offs between safety metrics and efficiency.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    percentiles = [r['percentile'] for r in percentile_results]
    accuracies = [r['accuracy'] * 100 for r in percentile_results]
    sensitivities = [r['sensitivity'] * 100 for r in percentile_results]
    specificities = [r['specificity'] * 100 for r in percentile_results]
    savings = [r['token_savings_pct'] for r in percentile_results]

    # Plot 1: Accuracy vs Percentile
    axes[0, 0].plot(percentiles, accuracies, marker='o', linewidth=2.5,
                    markersize=8, color='#2E86AB', label='Accuracy')
    axes[0, 0].set_xlabel('Confidence Percentile Threshold', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0, 0].set_title('Accuracy vs Percentile', fontweight='bold', fontsize=12)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Sensitivity vs Specificity
    axes[0, 1].plot(percentiles, sensitivities, marker='s', linewidth=2.5,
                    markersize=8, color='#D32F2F', label='Sensitivity (Catch Toxic)')
    axes[0, 1].plot(percentiles, specificities, marker='^', linewidth=2.5,
                    markersize=8, color='#06A77D', label='Specificity (Allow Safe)')
    axes[0, 1].set_xlabel('Confidence Percentile Threshold', fontweight='bold')
    axes[0, 1].set_ylabel('Rate (%)', fontweight='bold')
    axes[0, 1].set_title('Sensitivity vs Specificity', fontweight='bold', fontsize=12)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].axhline(y=100, color='gray', linestyle='--', alpha=0.3, label='Perfect')

    # Plot 3: Token Savings
    axes[1, 0].plot(percentiles, savings, marker='D', linewidth=2.5,
                    markersize=8, color='#06A77D')
    axes[1, 0].fill_between(percentiles, savings, alpha=0.3, color='#06A77D')
    axes[1, 0].set_xlabel('Confidence Percentile Threshold', fontweight='bold')
    axes[1, 0].set_ylabel('Token Savings (%)', fontweight='bold')
    axes[1, 0].set_title('Efficiency: Token Savings', fontweight='bold', fontsize=12)
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Safety-Efficiency Trade-off
    # Plot sensitivity on x-axis, savings on y-axis
    axes[1, 1].scatter(sensitivities, savings, s=150, c=percentiles,
                       cmap='RdYlGn_r', edgecolors='black', linewidth=1.5)
    for i, p in enumerate(percentiles):
        axes[1, 1].annotate(f'{p}%', (sensitivities[i], savings[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 1].set_xlabel('Sensitivity - Catch Rate for Toxic (%)', fontweight='bold')
    axes[1, 1].set_ylabel('Token Savings (%)', fontweight='bold')
    axes[1, 1].set_title('Safety-Efficiency Trade-off', fontweight='bold', fontsize=12)
    axes[1, 1].grid(alpha=0.3)

    # Add colorbar for percentile
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                               norm=plt.Normalize(vmin=min(percentiles), vmax=max(percentiles)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1, 1])
    cbar.set_label('Percentile', fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'percentile_safety_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create safety-focused visualizations"
    )
    parser.add_argument('--results-dir', required=True,
                        help='Path to experiment results')
    parser.add_argument('--percentile-analysis', required=True,
                        help='Path to percentile analysis JSON')
    parser.add_argument('--benchmark', default='toxicchat',
                        help='Benchmark name')
    parser.add_argument('--data-root', default='data',
                        help='Root data directory')
    parser.add_argument('--output', default='plots',
                        help='Output directory for plots')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    predictions_path = results_dir / 'predictions.jsonl'
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CREATING SAFETY VISUALIZATIONS")
    print("="*70)
    print()

    # Load data
    print("Loading data...")
    predictions, ground_truth = load_data(
        predictions_path,
        Path(args.data_root),
        args.benchmark
    )
    print(f"  {len(predictions):,} predictions")
    print()

    # Plot 1: 2x2 Confusion Matrix
    print("Creating 2x2 confusion matrix...")
    cm = plot_confusion_matrix_2x2(predictions, ground_truth, output_dir)
    print()

    # Plot 2: Percentile curves
    if Path(args.percentile_analysis).exists():
        print("Creating percentile safety curves...")
        with open(args.percentile_analysis) as f:
            percentile_data = json.load(f)
        plot_percentile_curves(percentile_data['results'], output_dir)
        print()

    print("="*70)
    print("✓ All visualizations created!")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
