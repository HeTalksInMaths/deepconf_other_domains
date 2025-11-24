"""
XSum dataset loader with hallucination annotations.

Loads:
- XSum dataset from HuggingFace
- Google hallucination annotations
- Handles both evaluation modes
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: datasets library not installed. Run: pip install datasets")


@dataclass
class HallucinatedSpan:
    """A hallucinated span in a summary."""
    span: str
    start: int
    end: int
    hallucination_type: str  # 'intrinsic' or 'extrinsic'


@dataclass
class XSumInstance:
    """A single XSum instance with optional hallucination annotations."""
    # Core data
    id: str
    document: str
    reference_summary: str
    
    # Generated summary (if evaluating a system)
    system_name: Optional[str] = None
    generated_summary: Optional[str] = None
    
    # Hallucination annotations (from Google dataset)
    hallucinated_spans: List[HallucinatedSpan] = field(default_factory=list)
    faithfulness_score: Optional[float] = None
    factuality_score: Optional[float] = None
    
    # Automatic metrics (if pre-computed)
    rouge_scores: Optional[Dict[str, float]] = None
    bertscore: Optional[float] = None
    alignscore: Optional[float] = None
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    @property
    def has_hallucinations(self) -> bool:
        """Check if this instance has any hallucinations."""
        return len(self.hallucinated_spans) > 0
    
    @property
    def hallucination_rate(self) -> float:
        """Proportion of summary that is hallucinated (by character)."""
        if not self.generated_summary:
            return 0.0
        total_chars = len(self.generated_summary)
        if total_chars == 0:
            return 0.0
        hallucinated_chars = sum(len(span.span) for span in self.hallucinated_spans)
        return hallucinated_chars / total_chars


class XSumHallucinationLoader:
    """Load XSum dataset with hallucination annotations."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize loader.
        
        Args:
            cache_dir: Directory to cache datasets
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.xsum_data = None
        self.hallucination_data = None
    
    def load_xsum(self, split: str = 'test') -> List[XSumInstance]:
        """
        Load XSum dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
        
        Returns:
            List of XSum instances
        """
        print(f"Loading XSum {split} split...")
        dataset = load_dataset('xsum', split=split, cache_dir=self.cache_dir)
        
        instances = []
        for idx, item in enumerate(dataset):
            instance = XSumInstance(
                id=f"xsum_{split}_{idx}",
                document=item['document'],
                reference_summary=item['summary']
            )
            instances.append(instance)
        
        print(f"Loaded {len(instances)} instances from XSum {split}")
        return instances
    
    def load_hallucination_annotations(self, 
                                      hallucination_file: str) -> List[XSumInstance]:
        """
        Load Google XSum hallucination annotations.
        
        The annotations file should be CSV with columns:
        - bbcid: Document ID
        - system: System name
        - summary: Generated summary
        - hallucination_type: 'intrinsic' or 'extrinsic'
        - hallucinated_span: The hallucinated text
        - hallucinated_span_start: Start index
        - hallucinated_span_end: End index
        - worker_id: Annotator ID
        
        Additionally, the scores file should have:
        - system_bbcid: System_DocID
        - Faithful: Faithfulness score
        - Factual: Factuality score
        - R1, R2, RL: ROUGE scores
        - BERTScore: BERTScore
        
        Args:
            hallucination_file: Path to annotations CSV
        
        Returns:
            List of XSum instances with hallucination annotations
        """
        print(f"Loading hallucination annotations from {hallucination_file}...")
        df = pd.read_csv(hallucination_file)
        
        # Group by document and system
        instances_dict = {}
        
        for _, row in df.iterrows():
            doc_id = row['bbcid']
            system = row['system']
            key = f"{system}_{doc_id}"
            
            if key not in instances_dict:
                instances_dict[key] = XSumInstance(
                    id=key,
                    document="",  # Will be filled from XSum dataset
                    reference_summary="",  # Will be filled from XSum dataset
                    system_name=system,
                    generated_summary=row['summary']
                )
            
            # Add hallucination span if present
            if pd.notna(row.get('hallucinated_span')):
                span = HallucinatedSpan(
                    span=row['hallucinated_span'],
                    start=int(row['hallucinated_span_start']),
                    end=int(row['hallucinated_span_end']),
                    hallucination_type=row['hallucination_type']
                )
                instances_dict[key].hallucinated_spans.append(span)
        
        print(f"Loaded {len(instances_dict)} instances with hallucination annotations")
        return list(instances_dict.values())
    
    def load_faithfulness_scores(self, scores_file: str,
                                instances: List[XSumInstance]) -> List[XSumInstance]:
        """
        Load pre-computed faithfulness and factuality scores.
        
        Args:
            scores_file: Path to scores CSV
            instances: List of instances to update
        
        Returns:
            Updated list of instances
        """
        print(f"Loading faithfulness scores from {scores_file}...")
        df = pd.read_csv(scores_file)
        
        # Create lookup dictionary
        instances_dict = {inst.id: inst for inst in instances}
        
        for _, row in df.iterrows():
            key = row['system_bbcid']
            if key in instances_dict:
                inst = instances_dict[key]
                inst.faithfulness_score = float(row['Faithful'])
                inst.factuality_score = float(row['Factual'])
                inst.rouge_scores = {
                    'R1': float(row['R1']),
                    'R2': float(row['R2']),
                    'RL': float(row['RL'])
                }
                if 'BERTScore' in row:
                    inst.bertscore = float(row['BERTScore'])
        
        return instances
    
    def merge_xsum_with_annotations(self,
                                    xsum_instances: List[XSumInstance],
                                    annotated_instances: List[XSumInstance]) -> List[XSumInstance]:
        """
        Merge XSum base data with hallucination annotations.
        
        Args:
            xsum_instances: Base XSum instances
            annotated_instances: Instances with hallucination annotations
        
        Returns:
            Merged instances
        """
        # Create lookup by BBC ID
        xsum_lookup = {}
        for inst in xsum_instances:
            # Extract BBC ID from XSum instance ID if available
            # Format is typically xsum_test_0, xsum_test_1, etc.
            xsum_lookup[inst.id] = inst
        
        # Update annotated instances with source documents
        merged = []
        for ann_inst in annotated_instances:
            # Try to find matching XSum instance
            # This requires knowing the mapping between BBC IDs and XSum indices
            # For now, keep annotated instance as-is
            merged.append(ann_inst)
        
        print(f"Merged {len(merged)} instances")
        return merged


def load_xsum_for_deepconf(split: str = 'test',
                          max_instances: Optional[int] = None) -> List[XSumInstance]:
    """
    Convenience function to load XSum for DeepConf experiments.
    
    Args:
        split: Dataset split
        max_instances: Maximum number of instances to load
    
    Returns:
        List of XSum instances
    """
    loader = XSumHallucinationLoader()
    instances = loader.load_xsum(split=split)
    
    if max_instances:
        instances = instances[:max_instances]
    
    return instances


def load_hallucination_benchmark(hallucination_file: str,
                                 scores_file: Optional[str] = None,
                                 max_instances: Optional[int] = None) -> List[XSumInstance]:
    """
    Load the Google XSum hallucination benchmark.
    
    Args:
        hallucination_file: Path to hallucination annotations CSV
        scores_file: Optional path to scores CSV
        max_instances: Maximum instances to load
    
    Returns:
        List of instances with hallucination annotations
    """
    loader = XSumHallucinationLoader()
    instances = loader.load_hallucination_annotations(hallucination_file)
    
    if scores_file:
        instances = loader.load_faithfulness_scores(scores_file, instances)
    
    if max_instances:
        instances = instances[:max_instances]
    
    return instances


# Example usage
if __name__ == "__main__":
    # Load base XSum dataset
    instances = load_xsum_for_deepconf(split='test', max_instances=10)
    print(f"Loaded {len(instances)} instances")
    
    # Example: Load with hallucination annotations
    # instances = load_hallucination_benchmark(
    #     hallucination_file='path/to/xsum_hallucination_annotations.csv',
    #     scores_file='path/to/xsum_faithfulness_scores.csv',
    #     max_instances=100
    # )
    
    for inst in instances[:3]:
        print(f"\nDocument ID: {inst.id}")
        print(f"Document: {inst.document[:200]}...")
        print(f"Reference: {inst.reference_summary}")
        if inst.has_hallucinations:
            print(f"Hallucinations: {len(inst.hallucinated_spans)}")
