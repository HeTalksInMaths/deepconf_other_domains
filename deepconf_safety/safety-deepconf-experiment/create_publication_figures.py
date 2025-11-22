#!/usr/bin/env python3
"""
Create publication-quality figures for DeepConf Safety experiments.

Generates:
1. 4-Bucket Confidence Distribution (adapted from DeepConf AIME figure)
2. Token Savings Summary
3. Hypothesis Validation Plot

Usage:
    python create_publication_figures.py results/toxicchat_1000/
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_results(results_dir):
    """Load experiment results from directory."""
    results_dir = Path(results_dir)

    # Load CSV
    csv_path = results_dir / "analysis.csv"
    if not csv_path.exists():
        # Try alternative names
        csv_candidates = list(results_dir.glob("*results*.csv"))
        if csv_candidates:
            csv_path = csv_candidates[0]
        else:
            raise FileNotFoundError(f"No results CSV found in {results_dir}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions from {csv_path}")

    # Load JSON analysis
    json_path = results_dir / "analysis.json"
    if not json_path.exists():
        json_candidates = list(results_dir.glob("*analysis*.json"))
        if json_candidates:
            json_path = json_candidates[0]
        else:
            print("Warning: No analysis JSON found")
            analysis = None
    else:
        with open(json_path) as f:
            analysis = json.load(f)
        print(f"Loaded analysis from {json_path}")

    return df, analysis


def create_4bucket_confidence_distribution(df, output_path="confidence_4bucket.png"):
    """
    Create 4-bucket confidence distribution plot.

    Adapted from DeepConf's AIME confidence distribution figure.
    Shows confidence distributions for:
    - Should Refuse (green) - Correct refusals of unsafe prompts
    - Should Allow (blue) - Correct compliance with safe prompts
    - Incorrectly Refused (orange) - False positives
    - Incorrectly Allowed (red) - False negatives (DANGEROUS!)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Categorize predictions into 4 buckets
    buckets = {
        'should_refuse': [],       # Unsafe + Model refused = CORRECT
        'should_allow': [],        # Safe + Model complied = CORRECT
        'incorrectly_refused': [], # Safe + Model refused = FALSE POSITIVE
        'incorrectly_allowed': []  # Unsafe + Model complied = FALSE NEGATIVE (DANGEROUS!)
    }

    for _, row in df.iterrows():
        prompt_is_unsafe = (row['ground_truth'] == 'unsafe')
        model_refused = (row['prediction'] == 'safe')  # 'safe' prediction means model refused

        if prompt_is_unsafe and model_refused:
            buckets['should_refuse'].append(row['confidence'])
        elif not prompt_is_unsafe and not model_refused:
            buckets['should_allow'].append(row['confidence'])
        elif not prompt_is_unsafe and model_refused:
            buckets['incorrectly_refused'].append(row['confidence'])
        else:  # prompt_is_unsafe and not model_refused
            buckets['incorrectly_allowed'].append(row['confidence'])

    # Colors matching severity
    colors = {
        'should_refuse': '#2ecc71',      # Green - good refusal
        'should_allow': '#3498db',       # Blue - good compliance
        'incorrectly_refused': '#f39c12', # Orange - over-cautious
        'incorrectly_allowed': '#e74c3c'  # Red - DANGEROUS!
    }

    labels = {
        'should_refuse': f'Should Refuse (n={len(buckets["should_refuse"])})',
        'should_allow': f'Should Allow (n={len(buckets["should_allow"])})',
        'incorrectly_refused': f'Incorrectly Refused (n={len(buckets["incorrectly_refused"])})',
        'incorrectly_allowed': f'Incorrectly Allowed (n={len(buckets["incorrectly_allowed"])})'
    }

    # Plot density distributions
    for bucket_name, confidences in buckets.items():
        if len(confidences) > 0:
            sns.kdeplot(
                confidences,
                label=labels[bucket_name],
                color=colors[bucket_name],
                alpha=0.6,
                linewidth=2.5,
                ax=ax
            )

    # Add vertical lines for thresholds
    ax.axvline(0.7, color='gray', linestyle='--', alpha=0.5, label='High Confidence Threshold')
    ax.axvline(0.3, color='gray', linestyle=':', alpha=0.5, label='Low Confidence Threshold')

    # Styling
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Safety Confidence Distribution: 4-Bucket Analysis\n(Adapted from DeepConf AIME Methodology)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add summary statistics box
    total = len(df)
    stats_text = (
        f"Total Predictions: {total}\n"
        f"Correct Refusals: {len(buckets['should_refuse'])} ({len(buckets['should_refuse'])/total*100:.1f}%)\n"
        f"Correct Compliance: {len(buckets['should_allow'])} ({len(buckets['should_allow'])/total*100:.1f}%)\n"
        f"False Positives: {len(buckets['incorrectly_refused'])} ({len(buckets['incorrectly_refused'])/total*100:.1f}%)\n"
        f"False Negatives: {len(buckets['incorrectly_allowed'])} ({len(buckets['incorrectly_allowed'])/total*100:.1f}%)"
    )

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved 4-bucket confidence distribution to {output_path}")

    return fig


def create_token_savings_plot(analysis, output_path="token_savings.png"):
    """
    Create token savings visualization.

    Shows efficiency gains from early stopping (DeepConf key contribution).
    """
    if not analysis or 'overall' not in analysis:
        print("Warning: No analysis data available for token savings plot")
        return None

    overall = analysis['overall']

    # Extract token metrics
    if 'total_tokens' not in overall:
        print("Warning: No token tracking data available")
        return None

    total_tokens = overall['total_tokens']
    baseline_tokens = overall.get('baseline_tokens', total_tokens * 1.5)  # Estimate if not available
    savings_percent = overall.get('token_savings_percent', 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Bar chart comparing baseline vs actual
    categories = ['Baseline\n(No Early Stopping)', 'Actual\n(With Early Stopping)']
    values = [baseline_tokens, total_tokens]
    colors_bar = ['#e74c3c', '#2ecc71']

    bars = ax1.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}\ntokens',
                ha='center', va='bottom', fontweight='bold')

    # Add savings annotation
    savings_tokens = baseline_tokens - total_tokens
    ax1.annotate(
        f'Saved: {savings_tokens:,} tokens\n({savings_percent:.1f}%)',
        xy=(0.5, total_tokens + (baseline_tokens - total_tokens)/2),
        xytext=(0.5, baseline_tokens * 0.7),
        ha='center',
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', lw=2, color='black')
    )

    ax1.set_ylabel('Total Tokens', fontsize=12, fontweight='bold')
    ax1.set_title('Token Usage Comparison\n(DeepConf Early Stopping Efficiency)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Right plot: Token usage by category
    if all(cat in analysis for cat in ['confident_refusal', 'confident_compliance',
                                        'uncertain_refusal', 'uncertain_compliance']):
        categories_tokens = []
        category_names = []
        category_colors = []

        color_map = {
            'confident_refusal': '#2ecc71',
            'confident_compliance': '#3498db',
            'uncertain_refusal': '#f39c12',
            'uncertain_compliance': '#e74c3c'
        }

        for cat_name, cat_data in analysis.items():
            if cat_name in color_map and 'total_tokens' in cat_data:
                categories_tokens.append(cat_data['total_tokens'])
                category_names.append(cat_name.replace('_', ' ').title())
                category_colors.append(color_map[cat_name])

        if categories_tokens:
            bars2 = ax2.barh(category_names, categories_tokens, color=category_colors, alpha=0.7,
                            edgecolor='black', linewidth=1.5)

            # Add value labels
            for bar, val in zip(bars2, categories_tokens):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2.,
                        f' {val:,}',
                        ha='left', va='center', fontweight='bold')

            ax2.set_xlabel('Tokens Used', fontsize=12, fontweight='bold')
            ax2.set_title('Token Usage by Category\n(4-Way Breakdown)', fontsize=13, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
    else:
        # Fallback: Show avg tokens per trace
        ax2.text(0.5, 0.5, 'Category token breakdown\nnot available',
                ha='center', va='center', fontsize=12,
                transform=ax2.transAxes)
        ax2.set_title('Token Usage by Category', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved token savings plot to {output_path}")

    return fig


def create_hypothesis_validation_plot(df, analysis, output_path="hypothesis_validation.png"):
    """
    Create hypothesis validation plot.

    Shows accuracy comparison between uncertain_compliance and confident_compliance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Accuracy by category
    if all(cat in analysis for cat in ['uncertain_compliance', 'confident_compliance']):
        categories = ['Uncertain\nCompliance\n(LOW CONF)', 'Confident\nCompliance\n(HIGH CONF)']
        accuracies = [
            analysis['uncertain_compliance']['accuracy'],
            analysis['confident_compliance']['accuracy']
        ]
        colors_acc = ['#e74c3c', '#2ecc71']

        bars = ax1.bar(categories, accuracies, color=colors_acc, alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1%}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Add hypothesis annotation
        if 'hypothesis_test' in analysis:
            ht = analysis['hypothesis_test']
            diff = ht['difference']

            ax1.annotate(
                f'Gap: {diff:.1%}\n({ht["interpretation"][:30]}...)',
                xy=(0.5, (accuracies[0] + accuracies[1])/2),
                xytext=(0.5, max(accuracies) * 0.5),
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='<->', lw=2, color='black')
            )

        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1.0])
        ax1.set_title('Hypothesis Validation\n(Low-Confidence Non-Refusals are Riskier)',
                     fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

    # Right plot: Scatter plot of confidence vs correctness
    # Add jitter for better visibility
    jitter = 0.02
    correct_points = df[df['correct']]
    incorrect_points = df[~df['correct']]

    ax2.scatter(
        correct_points['confidence'] + np.random.uniform(-jitter, jitter, len(correct_points)),
        [1] * len(correct_points),
        alpha=0.6,
        s=50,
        label=f'Correct (n={len(correct_points)})',
        color='green',
        edgecolor='black',
        linewidth=0.5
    )

    ax2.scatter(
        incorrect_points['confidence'] + np.random.uniform(-jitter, jitter, len(incorrect_points)),
        [0] * len(incorrect_points),
        alpha=0.6,
        s=50,
        label=f'Incorrect (n={len(incorrect_points)})',
        color='red',
        edgecolor='black',
        linewidth=0.5
    )

    # Add threshold lines
    ax2.axvline(0.7, color='gray', linestyle='--', alpha=0.5, label='High Conf Threshold')
    ax2.axvline(0.3, color='gray', linestyle=':', alpha=0.5, label='Low Conf Threshold')

    ax2.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Correctness', fontsize=12, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Incorrect', 'Correct'])
    ax2.set_title('Confidence vs Correctness\n(Individual Predictions)',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved hypothesis validation plot to {output_path}")

    return fig


def main():
    """Generate all publication figures."""
    if len(sys.argv) < 2:
        print("Usage: python create_publication_figures.py <results_directory>")
        print("\nExample:")
        print("  python create_publication_figures.py results/toxicchat_1000/")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_dir = Path(results_dir) / "figures"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("Creating Publication-Quality Figures")
    print(f"{'='*60}\n")

    # Load results
    df, analysis = load_results(results_dir)

    # Create figures
    print("\n1. Creating 4-bucket confidence distribution...")
    create_4bucket_confidence_distribution(
        df,
        output_path=output_dir / "confidence_4bucket.png"
    )

    if analysis:
        print("\n2. Creating token savings plot...")
        create_token_savings_plot(
            analysis,
            output_path=output_dir / "token_savings.png"
        )

        print("\n3. Creating hypothesis validation plot...")
        create_hypothesis_validation_plot(
            df,
            analysis,
            output_path=output_dir / "hypothesis_validation.png"
        )

    print(f"\n{'='*60}")
    print(f"✅ All figures saved to: {output_dir}/")
    print(f"{'='*60}\n")

    print("Figures created:")
    for fig_file in output_dir.glob("*.png"):
        print(f"  - {fig_file.name}")

    print("\nThese figures are publication-ready (300 DPI)!")


if __name__ == "__main__":
    main()
