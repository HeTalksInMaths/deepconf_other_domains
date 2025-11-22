"""
Summarization-specific confidence analysis for DeepConf adaptation.

Extends base confidence utilities with domain-specific logic for:
- Token-level hallucination detection
- Claim-level confidence
- Summary selection based on confidence
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

# Import base confidence utilities
import sys
sys.path.append('/home/claude/deepconf-adapter/scripts')
from confidence_utils import (
    TraceWithLogprobs,
    compute_token_confidence,
    compute_trace_confidence,
    compute_multi_trace_confidence
)


@dataclass
class ConfidenceSpan:
    """A span of tokens with associated confidence."""
    text: str
    start_idx: int
    end_idx: int
    confidence: float
    tokens: List[str]
    token_confidences: List[float]


@dataclass
class FactualClaim:
    """A factual claim extracted from a summary."""
    text: str
    confidence: float
    source_summary_idx: int  # Which summary this came from
    supporting_count: int = 1  # How many summaries support this claim


class SummarizationConfidenceAnalyzer:
    """Domain-specific confidence analysis for summarization tasks."""
    
    def __init__(self, low_confidence_threshold: float = 0.3,
                 high_confidence_threshold: float = 0.7):
        """
        Initialize analyzer.
        
        Args:
            low_confidence_threshold: Threshold below which confidence is considered low
            high_confidence_threshold: Threshold above which confidence is considered high
        """
        self.low_threshold = low_confidence_threshold
        self.high_threshold = high_confidence_threshold
    
    def compute_claim_confidence(self,
                                claim_text: str,
                                summary_text: str,
                                summary_tokens: List[str],
                                summary_logprobs: List[float]) -> float:
        """
        Compute confidence for a specific factual claim within a summary.
        
        Finds the claim in the summary and averages token confidence over that span.
        
        Args:
            claim_text: The factual claim text
            summary_text: Full summary text
            summary_tokens: Tokens from summary
            summary_logprobs: Log probabilities for each token
        
        Returns:
            Confidence score for the claim
        """
        # Find claim location in summary (simple substring match)
        claim_start = summary_text.find(claim_text)
        if claim_start == -1:
            # Claim not found, return average confidence
            return compute_token_confidence(summary_logprobs)
        
        claim_end = claim_start + len(claim_text)
        
        # Find corresponding token indices
        # This is approximate since tokenization may not align with characters
        char_to_token = self._build_char_to_token_mapping(summary_tokens, summary_text)
        
        token_start_idx = char_to_token.get(claim_start, 0)
        token_end_idx = char_to_token.get(claim_end, len(summary_tokens))
        
        # Extract logprobs for claim tokens
        claim_logprobs = summary_logprobs[token_start_idx:token_end_idx]
        
        if not claim_logprobs:
            return compute_token_confidence(summary_logprobs)
        
        return compute_token_confidence(claim_logprobs)
    
    def identify_low_confidence_spans(self,
                                     tokens: List[str],
                                     logprobs: List[float],
                                     window_size: int = 5) -> List[ConfidenceSpan]:
        """
        Identify spans with low generation confidence.
        
        Uses sliding window to find contiguous low-confidence regions.
        
        Args:
            tokens: List of tokens
            logprobs: Log probabilities for each token
            window_size: Size of sliding window
        
        Returns:
            List of low-confidence spans
        """
        if len(tokens) != len(logprobs):
            raise ValueError("Tokens and logprobs must have same length")
        
        # Compute per-token confidence
        token_confs = [-lp for lp in logprobs]
        
        # Find low-confidence windows
        low_conf_spans = []
        i = 0
        while i < len(tokens):
            # Check if window starting at i has low confidence
            window_end = min(i + window_size, len(tokens))
            window_conf = np.mean(token_confs[i:window_end])
            
            if window_conf < self.low_threshold:
                # Start of a low-confidence span
                span_start = i
                span_tokens = []
                span_confs = []
                
                # Extend span while confidence remains low
                while i < len(tokens) and token_confs[i] < self.low_threshold:
                    span_tokens.append(tokens[i])
                    span_confs.append(token_confs[i])
                    i += 1
                
                span = ConfidenceSpan(
                    text=' '.join(span_tokens),
                    start_idx=span_start,
                    end_idx=i,
                    confidence=np.mean(span_confs),
                    tokens=span_tokens,
                    token_confidences=span_confs
                )
                low_conf_spans.append(span)
            else:
                i += 1
        
        return low_conf_spans
    
    def extract_factual_claims(self, summary: str) -> List[str]:
        """
        Extract factual claims from a summary.
        
        Simple heuristic: Split by sentence boundaries and filter.
        More sophisticated: Use NER, dependency parsing, etc.
        
        Args:
            summary: Summary text
        
        Returns:
            List of factual claims
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # For single-sentence XSum summaries, extract noun phrases
        # or just use the whole summary
        if len(sentences) == 1:
            # For now, just use the whole summary as one claim
            # In practice, you'd want to extract sub-claims
            return sentences
        
        return sentences
    
    def compute_consensus_claims(self,
                                summaries: List[TraceWithLogprobs],
                                min_confidence: float = 0.5,
                                min_support: int = 2) -> List[FactualClaim]:
        """
        Extract high-confidence consensus claims across multiple summaries.
        
        Args:
            summaries: List of generated summaries with logprobs
            min_confidence: Minimum confidence for a claim to be included
            min_support: Minimum number of summaries that must support a claim
        
        Returns:
            List of consensus claims
        """
        all_claims = []
        
        # Extract claims from each summary
        for idx, summary in enumerate(summaries):
            claims = self.extract_factual_claims(summary.text)
            
            for claim_text in claims:
                claim_conf = self.compute_claim_confidence(
                    claim_text=claim_text,
                    summary_text=summary.text,
                    summary_tokens=summary.tokens,
                    summary_logprobs=summary.logprobs
                )
                
                claim = FactualClaim(
                    text=claim_text,
                    confidence=claim_conf,
                    source_summary_idx=idx
                )
                all_claims.append(claim)
        
        # Cluster similar claims
        claim_clusters = self._cluster_similar_claims(all_claims)
        
        # Filter by confidence and support
        consensus_claims = []
        for cluster in claim_clusters:
            avg_confidence = np.mean([c.confidence for c in cluster])
            if avg_confidence >= min_confidence and len(cluster) >= min_support:
                # Use highest-confidence variant
                best_claim = max(cluster, key=lambda c: c.confidence)
                best_claim.supporting_count = len(cluster)
                consensus_claims.append(best_claim)
        
        return consensus_claims
    
    def analyze_summary_quality_hierarchy(self,
                                         summary: TraceWithLogprobs,
                                         article: str,
                                         hallucination_detector: Optional[callable] = None,
                                         factual_consistency_fn: Optional[callable] = None) -> Dict[str, Any]:
        """
        Analyze hierarchical relationship: token confidence → hallucinations → consistency.
        
        Args:
            summary: Generated summary with logprobs
            article: Source article
            hallucination_detector: Function to detect hallucinations
            factual_consistency_fn: Function to compute factual consistency
        
        Returns:
            Dictionary with hierarchical analysis
        """
        # Level 1: Token-level confidence
        token_conf = compute_trace_confidence(summary)
        low_conf_spans = self.identify_low_confidence_spans(
            summary.tokens,
            summary.logprobs
        )
        
        # Level 2: Hallucination detection
        hallucination_info = {}
        if hallucination_detector:
            hallucination_info = hallucination_detector(article, summary.text)
        
        # Level 3: Factual consistency
        factual_consistency = None
        if factual_consistency_fn:
            factual_consistency = factual_consistency_fn(article, summary.text)
        
        return {
            'token_confidence': token_conf,
            'low_confidence_spans': low_conf_spans,
            'num_low_conf_spans': len(low_conf_spans),
            'avg_low_conf_span_confidence': np.mean([s.confidence for s in low_conf_spans]) if low_conf_spans else 1.0,
            'hallucination_info': hallucination_info,
            'factual_consistency': factual_consistency
        }
    
    def select_best_summary(self,
                           candidates: List[TraceWithLogprobs],
                           selection_method: str = 'confidence') -> Tuple[int, TraceWithLogprobs]:
        """
        Select best summary from multiple candidates.
        
        Args:
            candidates: List of candidate summaries
            selection_method: How to select
                - 'confidence': Highest average confidence
                - 'agreement': Most similar to others
                - 'min_low_conf': Fewest low-confidence tokens
        
        Returns:
            Tuple of (selected_index, selected_summary)
        """
        if not candidates:
            raise ValueError("No candidates provided")
        
        if selection_method == 'confidence':
            confidences = [compute_trace_confidence(c) for c in candidates]
            best_idx = int(np.argmax(confidences))
        
        elif selection_method == 'min_low_conf':
            low_conf_counts = []
            for cand in candidates:
                token_confs = [-lp for lp in cand.logprobs]
                low_count = sum(1 for conf in token_confs if conf < self.low_threshold)
                low_conf_counts.append(low_count)
            best_idx = int(np.argmin(low_conf_counts))
        
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        return best_idx, candidates[best_idx]
    
    def _build_char_to_token_mapping(self,
                                    tokens: List[str],
                                    text: str) -> Dict[int, int]:
        """
        Build mapping from character indices to token indices.
        
        This is approximate since tokenization may not align perfectly.
        """
        char_to_token = {}
        char_pos = 0
        
        for token_idx, token in enumerate(tokens):
            # Find this token in the text
            token_pos = text.find(token, char_pos)
            if token_pos != -1:
                for i in range(token_pos, token_pos + len(token)):
                    char_to_token[i] = token_idx
                char_pos = token_pos + len(token)
        
        return char_to_token
    
    def _cluster_similar_claims(self,
                               claims: List[FactualClaim],
                               similarity_threshold: float = 0.8) -> List[List[FactualClaim]]:
        """
        Cluster similar claims together.
        
        Uses simple string similarity for now.
        In practice, use semantic similarity.
        """
        if not claims:
            return []
        
        clusters = []
        used = set()
        
        for i, claim1 in enumerate(claims):
            if i in used:
                continue
            
            cluster = [claim1]
            used.add(i)
            
            for j, claim2 in enumerate(claims[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Simple similarity: character overlap
                similarity = self._string_similarity(claim1.text, claim2.text)
                
                if similarity >= similarity_threshold:
                    cluster.append(claim2)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity based on character overlap."""
        s1_set = set(s1.lower().split())
        s2_set = set(s2.lower().split())
        
        if not s1_set or not s2_set:
            return 0.0
        
        intersection = len(s1_set & s2_set)
        union = len(s1_set | s2_set)
        
        return intersection / union if union > 0 else 0.0


def analyze_token_hallucination_correlation(
    generated_summary: TraceWithLogprobs,
    hallucinated_spans: List[Tuple[int, int]]) -> Dict[str, Any]:
    """
    Analyze correlation between token confidence and hallucinated spans.
    
    Args:
        generated_summary: Summary with token logprobs
        hallucinated_spans: List of (start_idx, end_idx) for hallucinated spans
    
    Returns:
        Analysis dictionary with metrics
    """
    token_confs = [-lp for lp in generated_summary.logprobs]
    
    # Mark which tokens are in hallucinated spans
    is_hallucinated = [False] * len(token_confs)
    for start, end in hallucinated_spans:
        for i in range(start, min(end, len(is_hallucinated))):
            is_hallucinated[i] = True
    
    # Compute statistics
    halluc_confs = [conf for conf, is_hal in zip(token_confs, is_hallucinated) if is_hal]
    normal_confs = [conf for conf, is_hal in zip(token_confs, is_hallucinated) if not is_hal]
    
    return {
        'avg_halluc_confidence': np.mean(halluc_confs) if halluc_confs else 0.0,
        'avg_normal_confidence': np.mean(normal_confs) if normal_confs else 0.0,
        'confidence_difference': (np.mean(normal_confs) - np.mean(halluc_confs)) if halluc_confs and normal_confs else 0.0,
        'num_halluc_tokens': len(halluc_confs),
        'num_normal_tokens': len(normal_confs),
        'hallucination_rate': len(halluc_confs) / len(token_confs) if token_confs else 0.0
    }


# Example usage
if __name__ == "__main__":
    # Create example summary with logprobs
    summary = TraceWithLogprobs(
        text="The company announced record profits in Q3.",
        tokens=["The", "company", "announced", "record", "profits", "in", "Q3", "."],
        logprobs=[-0.5, -0.3, -0.2, -1.2, -1.5, -0.1, -0.9, -0.05]
    )
    
    analyzer = SummarizationConfidenceAnalyzer()
    
    # Identify low-confidence spans
    low_conf_spans = analyzer.identify_low_confidence_spans(
        summary.tokens,
        summary.logprobs
    )
    
    print(f"Found {len(low_conf_spans)} low-confidence spans:")
    for span in low_conf_spans:
        print(f"  '{span.text}' (confidence: {span.confidence:.3f})")
