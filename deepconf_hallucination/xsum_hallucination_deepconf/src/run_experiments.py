"""
Main experiment runner for XSum hallucination detection with DeepConf.

Implements all experimental designs from INSTANTIATION.md:
- Design A: Token-level hallucination detection
- Design B: Confidence-weighted claim extraction
- Design C: Hierarchical quality analysis
- Design D: Confidence calibration
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from xsum_loader import XSumInstance, load_xsum_for_deepconf
from summarization_confidence import (
    SummarizationConfidenceAnalyzer,
    TraceWithLogprobs,
    analyze_token_hallucination_correlation
)

# Add base confidence utilities
import sys
sys.path.append('/home/claude/deepconf-adapter/scripts')
from confidence_utils import compute_trace_confidence


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    experiment_name: str
    instance_id: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


class XSumHallucinationExperiment:
    """Main experiment class for XSum hallucination detection."""
    
    def __init__(self,
                 model_generator: Callable,
                 output_dir: str = "./results",
                 random_seed: int = 42):
        """
        Initialize experiment.
        
        Args:
            model_generator: Function that generates summaries with logprobs
                             Signature: (article: str, n_traces: int) -> List[TraceWithLogprobs]
            output_dir: Directory to save results
            random_seed: Random seed for reproducibility
        """
        self.model_generator = model_generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        self.analyzer = SummarizationConfidenceAnalyzer()
    
    def run_experiment_a_token_hallucination(self,
                                            instances: List[XSumInstance],
                                            factual_consistency_fn: Optional[Callable] = None) -> List[ExperimentResult]:
        """
        Design A: Token-level hallucination detection.
        
        Tests H1: Do low-confidence tokens correlate with hallucinated spans?
        
        Args:
            instances: List of XSum instances with hallucination annotations
            factual_consistency_fn: Optional function to compute AlignScore
        
        Returns:
            List of experiment results
        """
        print("Running Experiment A: Token-Level Hallucination Detection")
        results = []
        
        for instance in tqdm(instances):
            # Generate summary with logprobs
            traces = self.model_generator(instance.document, n_traces=1)
            if not traces:
                continue
            
            summary = traces[0]
            
            # Identify low-confidence spans
            low_conf_spans = self.analyzer.identify_low_confidence_spans(
                summary.tokens,
                summary.logprobs
            )
            
            # Compute token confidence
            token_conf = compute_trace_confidence(summary)
            
            # Analyze correlation with hallucinations (if available)
            correlation_metrics = {}
            if instance.hallucinated_spans:
                halluc_spans_indices = [
                    (span.start, span.end) 
                    for span in instance.hallucinated_spans
                ]
                correlation_metrics = analyze_token_hallucination_correlation(
                    summary,
                    halluc_spans_indices
                )
            
            # Compute factual consistency
            factual_consistency = None
            if factual_consistency_fn:
                factual_consistency = factual_consistency_fn(
                    instance.document,
                    summary.text
                )
            
            # Compile metrics
            metrics = {
                'token_confidence': token_conf,
                'num_low_conf_spans': len(low_conf_spans),
                'low_conf_span_chars': sum(len(s.text) for s in low_conf_spans),
                'summary_length': len(summary.text),
                'low_conf_proportion': sum(len(s.text) for s in low_conf_spans) / len(summary.text) if summary.text else 0.0,
                'has_annotations': bool(instance.hallucinated_spans),
                'num_hallucinated_spans': len(instance.hallucinated_spans),
                'factual_consistency': factual_consistency,
                **correlation_metrics
            }
            
            result = ExperimentResult(
                experiment_name='token_hallucination_detection',
                instance_id=instance.id,
                metrics=metrics,
                metadata={
                    'summary': summary.text,
                    'low_conf_spans': [{'text': s.text, 'conf': s.confidence} for s in low_conf_spans],
                    'hallucinated_spans': [asdict(s) for s in instance.hallucinated_spans]
                }
            )
            results.append(result)
        
        # Save results
        self._save_results(results, 'experiment_a_token_hallucination.json')
        
        return results
    
    def run_experiment_b_multi_trace_consensus(self,
                                              instances: List[XSumInstance],
                                              n_traces_list: List[int] = [1, 3, 5, 10],
                                              factual_consistency_fn: Optional[Callable] = None) -> List[ExperimentResult]:
        """
        Design B: Confidence-weighted claim extraction with multiple traces.
        
        Tests H2: Do multiple traces with consensus reduce hallucinations?
        Tests H4: Can confidence guide summary selection?
        
        Args:
            instances: List of XSum instances
            n_traces_list: List of trace counts to test
            factual_consistency_fn: Function to compute factual consistency
        
        Returns:
            List of experiment results
        """
        print("Running Experiment B: Multi-Trace Consensus")
        results = []
        
        for instance in tqdm(instances):
            for n_traces in n_traces_list:
                # Generate N summaries
                traces = self.model_generator(instance.document, n_traces=n_traces)
                if not traces:
                    continue
                
                # Approach 1: Single trace (baseline)
                single_trace = traces[0]
                single_conf = compute_trace_confidence(single_trace)
                single_factual = factual_consistency_fn(instance.document, single_trace.text) if factual_consistency_fn else None
                
                # Approach 2: Select best by confidence
                best_idx, best_trace = self.analyzer.select_best_summary(
                    traces,
                    selection_method='confidence'
                )
                best_conf = compute_trace_confidence(best_trace)
                best_factual = factual_consistency_fn(instance.document, best_trace.text) if factual_consistency_fn else None
                
                # Approach 3: Consensus claims (if multiple traces)
                if n_traces > 1:
                    consensus_claims = self.analyzer.compute_consensus_claims(
                        traces,
                        min_confidence=0.5,
                        min_support=max(2, n_traces // 3)
                    )
                    consensus_summary = ' '.join([c.text for c in consensus_claims])
                    consensus_conf = np.mean([c.confidence for c in consensus_claims]) if consensus_claims else 0.0
                    consensus_factual = factual_consistency_fn(instance.document, consensus_summary) if factual_consistency_fn and consensus_summary else None
                else:
                    consensus_conf = None
                    consensus_factual = None
                
                # Compile metrics
                metrics = {
                    'n_traces': n_traces,
                    'single_confidence': single_conf,
                    'single_factual_consistency': single_factual,
                    'best_confidence': best_conf,
                    'best_factual_consistency': best_factual,
                    'best_selected_idx': best_idx,
                    'consensus_confidence': consensus_conf,
                    'consensus_factual_consistency': consensus_factual,
                    'num_consensus_claims': len(consensus_claims) if n_traces > 1 else 0
                }
                
                result = ExperimentResult(
                    experiment_name='multi_trace_consensus',
                    instance_id=f"{instance.id}_n{n_traces}",
                    metrics=metrics,
                    metadata={
                        'n_traces': n_traces,
                        'reference_summary': instance.reference_summary,
                        'single_summary': single_trace.text,
                        'best_summary': best_trace.text,
                        'consensus_claims': [{'text': c.text, 'conf': c.confidence, 'support': c.supporting_count} for c in consensus_claims] if n_traces > 1 else []
                    }
                )
                results.append(result)
        
        # Save results
        self._save_results(results, 'experiment_b_multi_trace_consensus.json')
        
        return results
    
    def run_experiment_c_hierarchical_analysis(self,
                                              instances: List[XSumInstance],
                                              factual_consistency_fn: Optional[Callable] = None,
                                              rouge_fn: Optional[Callable] = None) -> List[ExperimentResult]:
        """
        Design C: Hierarchical quality analysis.
        
        Tests H3: Is there a hierarchical relationship?
        Token confidence → Hallucinations → Factual consistency → Overall quality
        
        Args:
            instances: List of XSum instances
            factual_consistency_fn: Function to compute factual consistency
            rouge_fn: Function to compute ROUGE scores
        
        Returns:
            List of experiment results
        """
        print("Running Experiment C: Hierarchical Quality Analysis")
        results = []
        
        for instance in tqdm(instances):
            # Generate summary
            traces = self.model_generator(instance.document, n_traces=1)
            if not traces:
                continue
            
            summary = traces[0]
            
            # Level 1: Token confidence
            token_conf = compute_trace_confidence(summary)
            low_conf_spans = self.analyzer.identify_low_confidence_spans(
                summary.tokens,
                summary.logprobs
            )
            
            # Level 2: Hallucination detection
            hallucination_score = len(low_conf_spans) / len(summary.tokens) if summary.tokens else 0.0
            
            # Level 3: Factual consistency
            factual_consistency = factual_consistency_fn(instance.document, summary.text) if factual_consistency_fn else None
            
            # Level 4: Overall quality (ROUGE)
            rouge_scores = rouge_fn(summary.text, instance.reference_summary) if rouge_fn else None
            
            # Compile metrics
            metrics = {
                'level1_token_confidence': token_conf,
                'level1_num_low_conf_spans': len(low_conf_spans),
                'level2_hallucination_score': hallucination_score,
                'level3_factual_consistency': factual_consistency,
                'level4_rouge_l': rouge_scores['rougeL'] if rouge_scores else None,
                'level4_rouge_1': rouge_scores['rouge1'] if rouge_scores else None,
                'level4_rouge_2': rouge_scores['rouge2'] if rouge_scores else None
            }
            
            result = ExperimentResult(
                experiment_name='hierarchical_quality_analysis',
                instance_id=instance.id,
                metrics=metrics,
                metadata={
                    'summary': summary.text,
                    'reference': instance.reference_summary
                }
            )
            results.append(result)
        
        # Save results
        self._save_results(results, 'experiment_c_hierarchical_analysis.json')
        
        # Perform correlation analysis
        self._analyze_hierarchical_correlations(results)
        
        return results
    
    def run_experiment_d_calibration_analysis(self,
                                             instances: List[XSumInstance],
                                             factual_consistency_fn: Callable,
                                             n_traces: int = 10) -> List[ExperimentResult]:
        """
        Design D: Confidence calibration analysis.
        
        Tests: Is confidence well-calibrated with factual consistency?
        
        Args:
            instances: List of XSum instances
            factual_consistency_fn: Function to compute factual consistency
            n_traces: Number of traces to generate per instance
        
        Returns:
            List of experiment results
        """
        print("Running Experiment D: Confidence Calibration Analysis")
        results = []
        
        for instance in tqdm(instances):
            # Generate multiple summaries to get confidence distribution
            traces = self.model_generator(instance.document, n_traces=n_traces)
            if not traces:
                continue
            
            for trace_idx, trace in enumerate(traces):
                # Compute confidence
                confidence = compute_trace_confidence(trace)
                
                # Compute actual quality
                factual_consistency = factual_consistency_fn(instance.document, trace.text)
                
                # Compile metrics
                metrics = {
                    'confidence': confidence,
                    'factual_consistency': factual_consistency,
                    'trace_idx': trace_idx
                }
                
                result = ExperimentResult(
                    experiment_name='calibration_analysis',
                    instance_id=f"{instance.id}_trace{trace_idx}",
                    metrics=metrics,
                    metadata={'summary': trace.text}
                )
                results.append(result)
        
        # Save results
        self._save_results(results, 'experiment_d_calibration_analysis.json')
        
        # Create calibration plot
        self._plot_calibration_curve(results)
        
        return results
    
    def _save_results(self, results: List[ExperimentResult], filename: str):
        """Save results to JSON file."""
        output_path = self.output_dir / filename
        
        results_dict = [
            {
                'experiment_name': r.experiment_name,
                'instance_id': r.instance_id,
                'metrics': r.metrics,
                'metadata': r.metadata
            }
            for r in results
        ]
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def _analyze_hierarchical_correlations(self, results: List[ExperimentResult]):
        """Analyze correlations in hierarchical analysis results."""
        # Extract metrics
        metrics_data = {
            'token_confidence': [],
            'hallucination_score': [],
            'factual_consistency': [],
            'rouge_l': []
        }
        
        for result in results:
            if result.metrics.get('level3_factual_consistency') is not None:
                metrics_data['token_confidence'].append(result.metrics['level1_token_confidence'])
                metrics_data['hallucination_score'].append(result.metrics['level2_hallucination_score'])
                metrics_data['factual_consistency'].append(result.metrics['level3_factual_consistency'])
                if result.metrics.get('level4_rouge_l') is not None:
                    metrics_data['rouge_l'].append(result.metrics['level4_rouge_l'])
        
        # Compute correlations
        from scipy.stats import pearsonr, spearmanr
        
        correlations = {}
        pairs = [
            ('token_confidence', 'hallucination_score'),
            ('token_confidence', 'factual_consistency'),
            ('hallucination_score', 'factual_consistency'),
            ('factual_consistency', 'rouge_l')
        ]
        
        print("\n=== Hierarchical Correlations ===")
        for var1, var2 in pairs:
            if metrics_data[var1] and metrics_data[var2] and len(metrics_data[var1]) == len(metrics_data[var2]):
                r, p = pearsonr(metrics_data[var1], metrics_data[var2])
                rho, p_spearman = spearmanr(metrics_data[var1], metrics_data[var2])
                print(f"{var1} <-> {var2}:")
                print(f"  Pearson r={r:.3f}, p={p:.4f}")
                print(f"  Spearman ρ={rho:.3f}, p={p_spearman:.4f}")
                correlations[f"{var1}_{var2}"] = {'pearson': r, 'spearman': rho}
        
        # Save correlations
        with open(self.output_dir / 'hierarchical_correlations.json', 'w') as f:
            json.dump(correlations, f, indent=2)
    
    def _plot_calibration_curve(self, results: List[ExperimentResult]):
        """Plot confidence calibration curve."""
        confidences = [r.metrics['confidence'] for r in results]
        factual_scores = [r.metrics['factual_consistency'] for r in results]
        
        # Bin by confidence
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        binned_factual = []
        for i in range(len(bins) - 1):
            mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i+1])
            if mask.sum() > 0:
                binned_factual.append(np.mean(np.array(factual_scores)[mask]))
            else:
                binned_factual.append(np.nan)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers, binned_factual, 'o-', label='Actual')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Factual Consistency')
        plt.title('Confidence Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
        print(f"Calibration plot saved to {self.output_dir / 'calibration_curve.png'}")


# Example mock model generator for testing
def mock_model_generator(article: str, n_traces: int = 1) -> List[TraceWithLogprobs]:
    """Mock model for testing (replace with actual model)."""
    summaries = []
    for _ in range(n_traces):
        # Generate mock summary
        summary_text = f"This is a mock summary of the article. It contains key points."
        tokens = summary_text.split()
        logprobs = np.random.randn(len(tokens)) * 0.5 - 0.3
        
        summaries.append(TraceWithLogprobs(
            text=summary_text,
            tokens=tokens,
            logprobs=logprobs.tolist()
        ))
    
    return summaries


if __name__ == "__main__":
    # Example usage
    print("XSum Hallucination Detection Experiments")
    print("=" * 50)
    
    # Load data
    instances = load_xsum_for_deepconf(split='test', max_instances=10)
    
    # Initialize experiment
    experiment = XSumHallucinationExperiment(
        model_generator=mock_model_generator,
        output_dir="./results/xsum_hallucination"
    )
    
    # Run experiments
    print("\nNote: Using mock model generator. Replace with actual model for real experiments.")
    
    # Experiment A: Token-level hallucination detection
    results_a = experiment.run_experiment_a_token_hallucination(instances)
    print(f"Experiment A completed: {len(results_a)} results")
    
    # Experiment B: Multi-trace consensus (comment out for quick test)
    # results_b = experiment.run_experiment_b_multi_trace_consensus(instances)
    # print(f"Experiment B completed: {len(results_b)} results")
