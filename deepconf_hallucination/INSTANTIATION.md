# XSum Hallucination Detection with DeepConf

**Domain**: Abstractive Summarization with Factual Consistency  
**Task**: Detect and reduce hallucinations in single-sentence summaries  
**Base Skill**: deepconf-adapter  

## 1. Domain Characterization

### What is the ground truth signal?

**Primary Signal**: Factual consistency between summary and source document
- **Hallucination annotations**: Human-labeled spans that are not supported by source
- **Automatic metrics**: AlignScore, QuestEval, BERTScore
- **Multi-dimensional**: 
  - Intrinsic hallucinations (contradicts source)
  - Extrinsic hallucinations (unsupported by source)

### What does "confidence" mean in summarization?

**Multiple interpretations possible:**

1. **Token-level confidence** = "I generated the right word"
   - Hypothesis: Low-confidence tokens are hallucinated facts
   - Signal: Per-token log probabilities

2. **Claim-level confidence** = "This factual claim is grounded"
   - Hypothesis: Low-confidence claims are factually inconsistent
   - Signal: Aggregate token confidence per claim

3. **Summary-level confidence** = "This is a good summary"
   - Hypothesis: Low confidence predicts poor overall quality
   - Signal: Aggregate across entire summary

**Hierarchical structure:**
```
Summary Quality (ROUGE, Human Rating)
    ↑
Factual Consistency (AlignScore, QuestEval)
    ↑
Token-Level Confidence (Logprobs)
```

### Cost structure

- **False negatives (missed hallucinations)**: High cost - misleading information
- **False positives (over-conservative)**: Medium cost - less informative summaries
- **Computational cost**: XSum summaries are short (1 sentence), so multiple traces are cheap

**Risk tolerance**: Conservative - prefer to flag potential hallucinations

## 2. Benchmark Structure Analysis

### XSum Dataset Structure

**Basic structure:**
- **Input**: News article (400 token limit)
- **Output**: Single-sentence summary (90 token limit)
- **Evaluation**: ROUGE, factual consistency metrics

**Google Hallucination Annotations:**
- 500 documents × 5 systems = 2,500 summaries
- Human-annotated hallucinated spans with types:
  - Intrinsic: Contradicts source
  - Extrinsic: Unsupported by source
- Faithfulness scores (0-1)
- Factuality binary labels

**Label structure**: Multi-dimensional
```python
{
    'summary': str,
    'hallucinated_spans': List[{'span': str, 'start': int, 'end': int, 'type': str}],
    'faithfulness_score': float,  # 0-1
    'factuality_score': float,     # 0-1
    'rouge_scores': {'R1': float, 'R2': float, 'RL': float},
    'bertscore': float,
    'alignscore': float
}
```

### Adaptation Strategy

Unlike MCQ problems, summaries are open-ended generation. We adapt DeepConf via:

**Strategy 1: Confidence-Weighted Claim Extraction** (Primary approach)
- Generate N summaries
- Extract factual claims from each
- Use confidence to weight claim validity
- Compose final summary from high-confidence consensus claims

**Strategy 2: Confidence as Quality Predictor** (Baseline)
- Generate N candidate summaries
- Compute confidence for each
- Select summary with highest confidence
- Test: Does high confidence → low hallucination rate?

**Strategy 3: Token-Level Hallucination Detection** (Diagnostic)
- Generate 1 summary with token logprobs
- Identify low-confidence tokens
- Correlate with hallucinated spans
- Test: Are low-confidence tokens more likely to be hallucinations?

## 3. Hypotheses

### H1: Token-Level Confidence Predicts Hallucinations
**Claim**: Tokens with low generation confidence are more likely to be part of hallucinated spans.

**Test**: 
- Generate summaries with token-level logprobs
- Correlate token confidence with Google's hallucination annotations
- Metric: Precision/Recall of low-confidence tokens as hallucination detector

**Expected result**: Low-confidence tokens should overlap with hallucinated spans

### H2: Multiple Traces Reduce Hallucination Rate
**Claim**: Generating multiple summaries and using confidence-weighted claim extraction reduces hallucinations compared to single-shot generation.

**Test**:
- Generate 10 summaries per document
- Extract claims, do confidence-weighted consensus
- Compare AlignScore and faithfulness vs. single-trace baseline

**Expected result**: Multi-trace approach should have higher faithfulness scores

### H3: Confidence Correlates Hierarchically with Quality
**Claim**: Low token confidence → hallucinations → low factual consistency → poor ROUGE/human ratings (hierarchical relationship).

**Test**:
- Compute token confidence, hallucination rate, AlignScore, ROUGE
- Test causal mediation: Does confidence → hallucinations → factual consistency?

**Expected result**: Significant mediation effect showing hierarchical structure

### H4: Confidence Enables Quality-Aware Summary Selection
**Claim**: Among N candidate summaries, the one with highest confidence has better factual consistency.

**Test**:
- Generate 10 candidates, rank by confidence
- Compare top-ranked vs. random selection on AlignScore

**Expected result**: Confidence ranking should outperform random selection

## 4. Experimental Design

### Design A: Token-Level Hallucination Detection (H1)

**Dataset**: Google XSum hallucination annotations (500 docs × 5 systems)

**Setup**:
```python
for doc_id, summary_system in hallucination_data:
    # Generate summary with logprobs
    summary, token_logprobs, tokens = model.generate(
        article, 
        return_logprobs=True
    )
    
    # Compute per-token confidence
    token_confidences = [-lp for lp in token_logprobs]
    
    # Get hallucination annotations for this summary
    hallucinated_spans = get_hallucinations(doc_id, summary_system)
    
    # Check overlap
    for token_idx, (token, conf) in enumerate(zip(tokens, token_confidences)):
        in_hallucination = is_token_in_spans(token_idx, hallucinated_spans)
        results.append({
            'token': token,
            'confidence': conf,
            'is_hallucinated': in_hallucination
        })
```

**Analysis**:
- Threshold analysis: At what confidence threshold do we best detect hallucinations?
- ROC curve: Confidence as hallucination predictor
- Stratify by: Intrinsic vs. extrinsic hallucinations

**Success criteria**: AUC > 0.65 for detecting hallucinated tokens

### Design B: Confidence-Weighted Claim Extraction (H2, H4)

**Dataset**: XSum test set (1,000 instances)

**Setup**:
```python
for article in xsum_test:
    # Generate N candidate summaries
    candidates = []
    for _ in range(10):
        summary, logprobs = model.generate(article, return_logprobs=True)
        confidence = compute_summary_confidence(logprobs)
        candidates.append({
            'summary': summary,
            'confidence': confidence,
            'logprobs': logprobs
        })
    
    # Strategy 1: Extract and vote on claims
    all_claims = []
    for cand in candidates:
        claims = extract_factual_claims(cand['summary'])
        for claim in claims:
            claim_conf = compute_claim_confidence(claim, cand['logprobs'])
            all_claims.append({'text': claim, 'confidence': claim_conf})
    
    # Cluster similar claims, keep high-confidence consensus
    consensus_claims = cluster_and_vote(all_claims, conf_threshold=0.6)
    final_summary = compose_summary(consensus_claims)
    
    # Strategy 2: Select highest-confidence candidate
    best_candidate = max(candidates, key=lambda x: x['confidence'])
    
    # Baselines
    single_shot = model.generate(article)  # No multi-trace
    random_candidate = random.choice(candidates)
    
    # Evaluate all approaches
    for name, summary in [
        ('single_shot', single_shot),
        ('consensus', final_summary),
        ('best_candidate', best_candidate['summary']),
        ('random_candidate', random_candidate['summary'])
    ]:
        results.append({
            'approach': name,
            'alignscore': compute_alignscore(article, summary),
            'rouge': compute_rouge(summary, reference),
            'faithfulness': evaluate_faithfulness(article, summary)
        })
```

**Analysis**:
- Compare: Consensus vs. Best-candidate vs. Single-shot vs. Random
- Efficiency: How many traces needed? (Ablation: 3, 5, 10, 20 traces)
- Stratify by: Article length, topic category

**Success criteria**: 
- Consensus approach improves AlignScore by ≥5% over single-shot
- Best-candidate outperforms random by ≥3%

### Design C: Hierarchical Quality Analysis (H3)

**Dataset**: XSum test set (1,000 instances)

**Setup**:
```python
for article in xsum_test:
    summary, logprobs, tokens = model.generate(article, return_logprobs=True)
    
    # Hierarchical measurements
    token_conf = compute_token_confidence(logprobs)
    
    # Level 1: Hallucination detection
    low_conf_spans = identify_low_confidence_spans(tokens, logprobs)
    hallucination_score = detect_hallucinations(article, summary)
    
    # Level 2: Factual consistency
    factual_consistency = compute_alignscore(article, summary)
    
    # Level 3: Overall quality
    rouge = compute_rouge(summary, reference)
    
    results.append({
        'token_confidence': token_conf,
        'hallucination_score': hallucination_score,
        'factual_consistency': factual_consistency,
        'rouge': rouge
    })
```

**Analysis**:
- Correlation matrix: All four levels
- Mediation analysis: Test if confidence → hallucinations → consistency → quality
- Path analysis: Quantify direct vs. indirect effects

**Success criteria**: Significant mediation showing hierarchical structure (p < 0.05)

### Design D: Confidence Calibration Analysis

**Dataset**: XSum test set (1,000 instances)

**Setup**: Generate summaries with varying confidence levels, measure actual quality

**Analysis**:
- Calibration plot: Confidence (x) vs. AlignScore (y)
- Stratify by confidence buckets: <0.3, 0.3-0.5, 0.5-0.7, >0.7
- Expected Calibration Error (ECE)

**Success criteria**: ECE < 0.15, positive correlation between confidence and quality

## 5. Implementation Components

### Required Tools

1. **XSum Loader**: Load dataset + hallucination annotations
2. **Model Interface**: Generate with logprobs (support multiple models)
3. **Claim Extractor**: Extract factual claims from summaries
4. **Claim Clusterer**: Group similar claims across traces
5. **Factual Consistency Evaluator**: AlignScore, QuestEval
6. **Baseline Metrics**: ROUGE, BERTScore

### Domain-Specific Confidence Utilities

```python
class SummarizationConfidenceAnalyzer:
    """Confidence analysis for abstractive summarization"""
    
    def compute_claim_confidence(self, claim: str, 
                                summary_logprobs: List[float],
                                summary_tokens: List[str]) -> float:
        """Compute confidence for a specific factual claim"""
        
    def identify_low_confidence_spans(self, 
                                     tokens: List[str],
                                     logprobs: List[float],
                                     threshold: float = 0.3) -> List[Span]:
        """Identify spans with low generation confidence"""
        
    def compute_consensus_claims(self,
                                summaries: List[str],
                                confidences: List[float]) -> List[str]:
        """Extract high-confidence consensus claims across summaries"""
```

## 6. Evaluation Metrics

### Primary Metrics
- **AlignScore**: Factual consistency (main metric)
- **Faithfulness Score**: From Google annotations
- **Hallucination Rate**: % of summaries with hallucinations

### Secondary Metrics
- **ROUGE-L**: Content overlap with reference
- **BERTScore**: Semantic similarity
- **QuestEval**: QA-based factual consistency

### Diagnostic Metrics
- **Token-level hallucination precision/recall**
- **Confidence calibration error**
- **Traces saved** (for early stopping variants)

### Stratification Dimensions
- Confidence buckets (<0.3, 0.3-0.7, >0.7)
- Article length (short, medium, long)
- Topic category (from XSum metadata)
- Hallucination type (intrinsic vs. extrinsic)

## 7. Expected Insights

### What we'll learn:

1. **Is token confidence predictive?**
   - Can we use logprobs to detect hallucinations?
   - What confidence threshold works best?

2. **Do multiple traces help?**
   - Does consensus reduce hallucinations?
   - What's the trace budget sweet spot?

3. **How does the hierarchy work?**
   - Is confidence → hallucinations → consistency → quality?
   - Which link is strongest/weakest?

4. **Can confidence guide generation?**
   - Should we generate more/fewer traces based on confidence?
   - Can we select better summaries using confidence?

### Boundary conditions:

- May not work for creative/opinion-based content
- Assumes hallucinations have lower confidence (may not always be true)
- Short summaries (1 sentence) may not have enough signal

## 8. Implementation Roadmap

**Phase 1**: Token-level hallucination detection (Design A)
- Validate basic hypothesis: Low confidence → Hallucinations
- Estimated time: 2-3 days

**Phase 2**: Multi-trace consensus (Design B)  
- Implement claim extraction and voting
- Compare against baselines
- Estimated time: 3-4 days

**Phase 3**: Hierarchical analysis (Design C)
- Full mediation analysis
- Path coefficients
- Estimated time: 2-3 days

**Phase 4**: Ablations and optimizations
- Confidence thresholds
- Number of traces
- Claim extraction methods
- Estimated time: 2-3 days

**Total estimated time**: 2-3 weeks for complete study

## 9. Success Criteria Summary

**Minimum viable result**:
- H1: Token confidence has AUC > 0.6 for hallucination detection
- H2: Multi-trace improves AlignScore by ≥3%

**Strong result**:
- H1: AUC > 0.7 for hallucination detection
- H2: Multi-trace improves AlignScore by ≥5%
- H3: Significant hierarchical mediation (p < 0.01)
- H4: Confidence-based selection beats random by ≥5%

**Publication-worthy result**:
- All above criteria met
- Generalizes across multiple models
- Confidence calibration ECE < 0.1
- Practical trace budget reduction (30%+ savings)
