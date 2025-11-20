"""
Confidence calculation utilities for DeepConf domain adaptation.

This module provides domain-agnostic confidence computation
based on token-level log probabilities.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TraceWithLogprobs:
    """Container for a single reasoning trace with log probabilities."""
    text: str
    logprobs: List[float]  # Log probability for each token
    tokens: List[str]
    metadata: Dict[str, Any] = None


def compute_token_confidence(logprobs: List[float], method: str = 'neg_avg_logprob', top_k: int = 5) -> float:
    """
    Compute confidence score from token log probabilities.
    
    Original DeepConf uses: negative average log probability of top-k tokens
    
    Args:
        logprobs: Log probabilities for each token
        method: Confidence computation method
            - 'neg_avg_logprob': Negative average log prob (DeepConf original)
            - 'entropy': Token-level entropy
            - 'min_prob': Minimum probability (most uncertain token)
        top_k: Number of top tokens to consider (for methods that use it)
    
    Returns:
        Confidence score (higher = more confident)
    """
    if not logprobs:
        return 0.0
    
    if method == 'neg_avg_logprob':
        # Original DeepConf: negative average log probability
        # More negative logprobs = less confident
        return -np.mean(logprobs)
    
    elif method == 'entropy':
        # Convert logprobs to probabilities and compute entropy
        probs = np.exp(logprobs)
        probs = probs / np.sum(probs)  # Normalize
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        # Normalize to 0-1 range (lower entropy = higher confidence)
        return 1.0 - (entropy / np.log(len(probs)))
    
    elif method == 'min_prob':
        # Use the minimum probability as uncertainty indicator
        min_logprob = np.min(logprobs)
        return -min_logprob  # Convert to confidence score
    
    else:
        raise ValueError(f"Unknown confidence method: {method}")


def compute_trace_confidence(trace: TraceWithLogprobs, method: str = 'neg_avg_logprob') -> float:
    """
    Compute confidence for a complete reasoning trace.
    
    Args:
        trace: Trace with log probabilities
        method: Confidence computation method
    
    Returns:
        Confidence score for the trace
    """
    return compute_token_confidence(trace.logprobs, method=method)


def compute_multi_trace_confidence(traces: List[TraceWithLogprobs], 
                                  aggregation: str = 'mean') -> float:
    """
    Compute aggregate confidence across multiple traces.
    
    Args:
        traces: List of traces with log probabilities
        aggregation: How to aggregate across traces
            - 'mean': Average confidence
            - 'min': Minimum confidence (most conservative)
            - 'max': Maximum confidence (most optimistic)
            - 'weighted': Weight by prediction agreement
    
    Returns:
        Aggregate confidence score
    """
    if not traces:
        return 0.0
    
    confidences = [compute_trace_confidence(t) for t in traces]
    
    if aggregation == 'mean':
        return np.mean(confidences)
    elif aggregation == 'min':
        return np.min(confidences)
    elif aggregation == 'max':
        return np.max(confidences)
    elif aggregation == 'weighted':
        # Weight by prediction agreement
        # This requires domain-specific prediction extraction
        raise NotImplementedError("Weighted aggregation requires domain-specific implementation")
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def compute_agreement_confidence(predictions: List[Any]) -> float:
    """
    Compute confidence based on prediction agreement across traces.
    
    Alternative to logprob-based confidence when logprobs unavailable.
    
    Args:
        predictions: List of predictions from different traces
    
    Returns:
        Agreement-based confidence (0-1)
    """
    if not predictions:
        return 0.0
    
    # Find most common prediction
    from collections import Counter
    vote_counts = Counter(predictions)
    most_common_count = vote_counts.most_common(1)[0][1]
    
    # Confidence = proportion agreeing with majority
    return most_common_count / len(predictions)


def should_generate_more_traces(current_confidence: float,
                               traces_so_far: int,
                               min_traces: int = 3,
                               max_traces: int = 20,
                               confidence_threshold: float = 0.7) -> bool:
    """
    Decide whether to generate additional reasoning traces.
    
    Early stopping logic for efficient inference.
    
    Args:
        current_confidence: Current aggregate confidence
        traces_so_far: Number of traces generated
        min_traces: Minimum traces before considering stopping
        max_traces: Maximum traces to generate
        confidence_threshold: Stop if confidence exceeds this
    
    Returns:
        True if should generate more traces, False if should stop
    """
    # Always generate minimum traces
    if traces_so_far < min_traces:
        return True
    
    # Never exceed maximum traces
    if traces_so_far >= max_traces:
        return False
    
    # Stop if confidence is high enough
    if current_confidence >= confidence_threshold:
        return False
    
    # Otherwise, generate more traces
    return True


def analyze_confidence_distribution(traces: List[TraceWithLogprobs],
                                   predictions: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Analyze confidence distribution across traces.
    
    Useful for understanding trace-level behavior.
    
    Args:
        traces: List of traces with log probabilities
        predictions: Optional list of predictions for agreement analysis
    
    Returns:
        Dictionary with confidence statistics
    """
    confidences = [compute_trace_confidence(t) for t in traces]
    
    analysis = {
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'confidence_range': np.max(confidences) - np.min(confidences),
        'num_traces': len(traces)
    }
    
    if predictions is not None:
        analysis['agreement_confidence'] = compute_agreement_confidence(predictions)
        
        # Check if high agreement correlates with high confidence
        analysis['high_agreement'] = analysis['agreement_confidence'] > 0.7
        analysis['high_confidence'] = analysis['mean_confidence'] > 0.7
        analysis['aligned'] = analysis['high_agreement'] == analysis['high_confidence']
    
    return analysis


# Example domain-specific extensions

class SafetyConfidenceAnalyzer:
    """Domain-specific confidence analysis for safety benchmarks."""
    
    @staticmethod
    def analyze_refusal_confidence(traces: List[TraceWithLogprobs],
                                   refusal_detector) -> Dict[str, Any]:
        """
        Analyze confidence specifically for refusal detection.
        
        Key insight: High-confidence refusals vs. low-confidence non-refusals
        """
        results = []
        for trace in traces:
            results.append({
                'confidence': compute_trace_confidence(trace),
                'refused': refusal_detector(trace.text),
                'trace_length': len(trace.tokens)
            })
        
        refusal_rate = np.mean([r['refused'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        # Categorize based on refusal + confidence
        if refusal_rate > 0.5 and avg_confidence > 0.7:
            category = 'confident_refusal'
        elif refusal_rate < 0.5 and avg_confidence < 0.3:
            category = 'uncertain_compliance'  # RISKY
        elif refusal_rate > 0.5 and avg_confidence < 0.3:
            category = 'uncertain_refusal'
        else:
            category = 'confident_compliance'
        
        return {
            'category': category,
            'refusal_rate': refusal_rate,
            'avg_confidence': avg_confidence,
            'risk_flag': category == 'uncertain_compliance'
        }


class ReasoningConfidenceAnalyzer:
    """Domain-specific confidence analysis for reasoning benchmarks."""
    
    @staticmethod
    def analyze_reasoning_progression(trace: TraceWithLogprobs,
                                     step_separator: str = '\n') -> Dict[str, Any]:
        """
        Analyze how confidence changes during reasoning steps.
        
        Can detect where reasoning goes wrong.
        """
        steps = trace.text.split(step_separator)
        tokens_per_step = len(trace.tokens) // len(steps)
        
        step_confidences = []
        for i in range(len(steps)):
            start_idx = i * tokens_per_step
            end_idx = (i + 1) * tokens_per_step
            step_logprobs = trace.logprobs[start_idx:end_idx]
            step_confidences.append(compute_token_confidence(step_logprobs))
        
        # Detect confidence drops
        confidence_drops = []
        for i in range(1, len(step_confidences)):
            if step_confidences[i] < step_confidences[i-1] * 0.7:  # 30% drop
                confidence_drops.append({
                    'step': i,
                    'drop_amount': step_confidences[i-1] - step_confidences[i]
                })
        
        return {
            'step_confidences': step_confidences,
            'confidence_drops': confidence_drops,
            'final_confidence': step_confidences[-1],
            'reasoning_consistency': np.std(step_confidences),
            'potential_error_locations': [d['step'] for d in confidence_drops]
        }
