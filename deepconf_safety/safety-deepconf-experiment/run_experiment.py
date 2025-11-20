"""
End-to-End Safety DeepConf Experiment with Qwen3-0.6B

Complete example showing how to:
1. Load a safety benchmark
2. Run DeepConf experiment with Qwen3-0.6B
3. Analyze results with confidence stratification
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from safety_deepconf import SafetyDeepConfExperiment, SafetyInstance
from benchmark_loaders import SyntheticSafetyBenchmark, SafetyBenchLoader
from qwen3_adapter import Qwen3SafetyAdapter


def run_safety_deepconf_experiment(
    model_name: str = "Qwen/Qwen3-0.6B",
    benchmark_name: str = "synthetic",
    num_instances: int = 20,
    min_traces: int = 3,
    max_traces: int = 10,
    early_stopping: bool = True,
    output_dir: str = "results/safety_experiment"
):
    """
    Run complete safety DeepConf experiment.
    
    Args:
        model_name: Qwen3 model to use
        benchmark_name: 'synthetic', 'toxicchat', 'harmbench', 'wildguard'
        num_instances: Number of instances (for synthetic only)
        min_traces: Minimum reasoning traces
        max_traces: Maximum reasoning traces
        early_stopping: Use confidence-based early stopping
        output_dir: Where to save results
    """
    
    print("=" * 60)
    print("SAFETY DEEPCONF EXPERIMENT")
    print("=" * 60)
    
    # Step 1: Load model
    print(f"\n[1/5] Loading model: {model_name}")
    model = Qwen3SafetyAdapter(model_name)
    print(f"✓ Model loaded")
    print(f"  Memory: {model.estimate_memory_usage()['memory_gb']:.2f} GB")
    
    # Step 2: Load benchmark
    print(f"\n[2/5] Loading benchmark: {benchmark_name}")
    if benchmark_name == "synthetic":
        instances = SyntheticSafetyBenchmark.create(num_instances=num_instances)
        print(f"✓ Created {len(instances)} synthetic instances")
    else:
        # For real benchmarks, you'd specify the path
        # instances = SafetyBenchLoader.load(benchmark_name, "path/to/data")
        raise NotImplementedError(f"Real benchmark loading: Download {benchmark_name} and specify path")
    
    print(f"  Total instances: {len(instances)}")
    print(f"  Unsafe: {sum(1 for i in instances if i.ground_truth_label == 'unsafe')}")
    print(f"  Safe: {sum(1 for i in instances if i.ground_truth_label == 'safe')}")
    
    # Step 3: Initialize experiment
    print(f"\n[3/5] Initializing experiment")
    experiment = SafetyDeepConfExperiment(
        min_traces=min_traces,
        max_traces=max_traces,
        confidence_threshold=0.7
    )
    print(f"✓ Experiment initialized")
    print(f"  Min traces: {min_traces}")
    print(f"  Max traces: {max_traces}")
    print(f"  Early stopping: {early_stopping}")
    
    # Step 4: Run experiment
    print(f"\n[4/5] Running experiment on {len(instances)} instances...")
    print("  (This may take a few minutes)")
    
    predictions = experiment.run_experiment(
        instances,
        model_callable=model,  # Qwen3Adapter is callable
        early_stopping=early_stopping,
        temperature=0.7,
        max_new_tokens=256
    )
    
    print(f"✓ Experiment complete!")
    print(f"  Total predictions: {len(predictions)}")
    
    # Step 5: Analyze results
    print(f"\n[5/5] Analyzing results...")
    analysis = experiment.analyze_results(predictions, instances)
    
    # Save results
    experiment.save_results(predictions, analysis, Path(output_dir))
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n[Overall Performance]")
    print(f"  Accuracy: {analysis['overall']['accuracy']:.3f}")
    print(f"  Avg traces used: {analysis['overall']['avg_traces']:.1f} / {max_traces}")
    print(f"  Avg confidence: {analysis['overall']['avg_confidence']:.3f}")
    
    print("\n[Performance by Confidence Category]")
    for category in ['confident_refusal', 'confident_compliance', 
                     'uncertain_refusal', 'uncertain_compliance']:
        if category in analysis:
            cat_data = analysis[category]
            print(f"\n  {category.replace('_', ' ').title()}:")
            print(f"    Count: {cat_data['count']}")
            print(f"    Accuracy: {cat_data['accuracy']:.3f}")
            print(f"    Avg confidence: {cat_data['avg_confidence']:.3f}")
            print(f"    Avg traces: {cat_data['avg_traces']:.1f}")
    
    if 'hypothesis_test' in analysis:
        print("\n[KEY FINDING: Hypothesis Test]")
        print("  Research Question: Are low-confidence non-refusals riskier?")
        print(f"  Uncertain compliance accuracy: {analysis['hypothesis_test']['uncertain_compliance_accuracy']:.3f}")
        print(f"  Confident compliance accuracy: {analysis['hypothesis_test']['confident_compliance_accuracy']:.3f}")
        print(f"  Difference: {analysis['hypothesis_test']['difference']:.3f}")
        if analysis['hypothesis_test']['difference'] > 0:
            print("  ✓ HYPOTHESIS SUPPORTED: Low-confidence non-refusals are indeed riskier!")
        else:
            print("  ✗ Hypothesis not supported in this experiment")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("\n" + "=" * 60)
    
    return predictions, analysis


def compare_baseline_vs_deepconf(
    model_name: str = "Qwen/Qwen3-0.6B",
    num_instances: int = 20
):
    """
    Compare single-trace baseline vs multi-trace DeepConf.
    
    This demonstrates the value of confidence estimation.
    """
    print("\n" + "=" * 60)
    print("BASELINE VS DEEPCONF COMPARISON")
    print("=" * 60)
    
    # Run baseline (single trace, n=1)
    print("\n[BASELINE] Single trace (n=1)")
    _, baseline_analysis = run_safety_deepconf_experiment(
        model_name=model_name,
        num_instances=num_instances,
        min_traces=1,
        max_traces=1,
        early_stopping=False,
        output_dir="results/baseline_n1"
    )
    
    # Run fixed multi-trace (n=5)
    print("\n\n[FIXED] Fixed 5 traces")
    _, fixed_analysis = run_safety_deepconf_experiment(
        model_name=model_name,
        num_instances=num_instances,
        min_traces=5,
        max_traces=5,
        early_stopping=False,
        output_dir="results/fixed_n5"
    )
    
    # Run adaptive DeepConf
    print("\n\n[DEEPCONF] Adaptive (3-10 traces, early stopping)")
    _, deepconf_analysis = run_safety_deepconf_experiment(
        model_name=model_name,
        num_instances=num_instances,
        min_traces=3,
        max_traces=10,
        early_stopping=True,
        output_dir="results/deepconf_adaptive"
    )
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print("\n| Method          | Accuracy | Avg Traces | Avg Confidence |")
    print("|-----------------|----------|------------|----------------|")
    print(f"| Baseline (n=1)  | {baseline_analysis['overall']['accuracy']:.3f}    | "
          f"{baseline_analysis['overall']['avg_traces']:.1f}        | "
          f"{baseline_analysis['overall']['avg_confidence']:.3f}          |")
    print(f"| Fixed (n=5)     | {fixed_analysis['overall']['accuracy']:.3f}    | "
          f"{fixed_analysis['overall']['avg_traces']:.1f}        | "
          f"{fixed_analysis['overall']['avg_confidence']:.3f}          |")
    print(f"| DeepConf (3-10) | {deepconf_analysis['overall']['accuracy']:.3f}    | "
          f"{deepconf_analysis['overall']['avg_traces']:.1f}        | "
          f"{deepconf_analysis['overall']['avg_confidence']:.3f}          |")
    
    # Efficiency analysis
    accuracy_gain = deepconf_analysis['overall']['accuracy'] - baseline_analysis['overall']['accuracy']
    traces_vs_fixed = deepconf_analysis['overall']['avg_traces'] / fixed_analysis['overall']['avg_traces']
    
    print(f"\n[Efficiency Analysis]")
    print(f"  Accuracy improvement over baseline: {accuracy_gain:+.3f}")
    print(f"  Traces used vs fixed n=5: {traces_vs_fixed:.1%}")
    print(f"  Interpretation: DeepConf achieves {accuracy_gain:+.1%} accuracy gain")
    print(f"  while using {traces_vs_fixed:.0%} of fixed-budget traces")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Safety DeepConf Experiment")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", 
                       help="Model name (e.g., Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B)")
    parser.add_argument("--benchmark", default="synthetic",
                       help="Benchmark name (synthetic, toxicchat, harmbench, wildguard)")
    parser.add_argument("--num-instances", type=int, default=20,
                       help="Number of instances (for synthetic)")
    parser.add_argument("--min-traces", type=int, default=3,
                       help="Minimum traces before early stopping")
    parser.add_argument("--max-traces", type=int, default=10,
                       help="Maximum traces to generate")
    parser.add_argument("--no-early-stopping", action="store_true",
                       help="Disable confidence-based early stopping")
    parser.add_argument("--compare", action="store_true",
                       help="Run baseline vs DeepConf comparison")
    parser.add_argument("--output", default="results/safety_experiment",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison experiment
        compare_baseline_vs_deepconf(
            model_name=args.model,
            num_instances=args.num_instances
        )
    else:
        # Run single experiment
        run_safety_deepconf_experiment(
            model_name=args.model,
            benchmark_name=args.benchmark,
            num_instances=args.num_instances,
            min_traces=args.min_traces,
            max_traces=args.max_traces,
            early_stopping=not args.no_early_stopping,
            output_dir=args.output
        )
