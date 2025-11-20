---
name: deepconf-adapter
description: "Adapt DeepConf's confidence-based early stopping framework from math problems to new domains (safety, quality, reasoning). Use when: (1) porting DeepConf to non-math benchmarks, (2) designing experiments with confidence estimation on subjective tasks, (3) understanding domain-specific confidence signal interpretation, (4) building multi-trace evaluation systems beyond verifiable correctness."
---

# DeepConf Domain Adapter

Adapt DeepConf's confidence estimation approach from verifiable math problems to other domains with thoughtful experimental design.

## Core DeepConf Mechanism (Domain-Agnostic)

DeepConf is a purely inference-time method with no training:
- Generate multiple reasoning traces in parallel
- Compute confidence from logprobs: `confidence = -mean(log_prob(top_k_tokens))`
- Apply early stopping when confidence drops below threshold
- Use majority voting for final answer

## Adaptation Workflow

Follow this sequence when porting DeepConf to a new domain:

### 1. Characterize the Domain

**Ask these questions before implementation:**

What is the ground truth signal?
- Math: Binary correctness, algorithmically verifiable
- Safety: Multi-class labels (safe/controversial/unsafe), human-annotated
- Quality: Continuous or comparative scores, often subjective

What does "confidence" mean here?
- Math: "I computed the right answer"
- Safety: Could mean "confident in safety assessment", "confident in refusal decision", or "confident about boundary"
- Quality: May not correlate with quality at all

What's the cost structure?
- False positive cost vs. false negative cost
- Computational cost of additional traces
- Risk tolerance for the application

**Action**: Read `references/domain_adaptation_framework.md` for detailed domain characterization checklist.

### 2. Analyze Benchmark Structure

**Understand the data format:**

Label structure?
- Binary: Direct majority voting
- Multi-class: Consider hierarchical decisions or per-class confidence
- Multi-label: Separate confidence per label
- Continuous: Use variance as confidence proxy

Key benchmark characteristics?
- Single correct answer or multiple valid approaches?
- Temporal dynamics (does quality degrade over generation)?
- Domain-specific categories to stratify by?

**Action**: Read `references/benchmark_structures.md` for benchmark-specific adaptation patterns.

**Example - Safety with Refusal:**
```python
# High-confidence refusals = appropriate boundary setting
# Low-confidence non-refusals = potential safety risk

def interpret_safety_confidence(traces, confidences):
    refusal_rate = sum(is_refusal(t) for t in traces) / len(traces)
    avg_conf = mean(confidences)
    
    if refusal_rate > 0.5 and avg_conf > 0.7:
        return "confident_refusal"  # Good!
    elif refusal_rate < 0.5 and avg_conf < 0.3:
        return "uncertain_compliance"  # FLAG THIS
    elif refusal_rate > 0.5 and avg_conf < 0.3:
        return "uncertain_refusal"
    else:
        return "confident_compliance"
```

### 3. Formulate Hypotheses

**Generate testable predictions:**

Template:
- H1: [Relationship between confidence and outcome]
- H2: [Benefit of multiple traces at different confidence levels]
- H3: [Domain-specific insight]

Example (Safety):
- H1: Low-confidence non-refusals have higher unsafe rates than high-confidence non-refusals
- H2: Multiple reasoning traces improve safety consistency for borderline cases
- H3: Confidence threshold can separate legitimate edge cases from alignment failures

**Action**: Read `references/experimental_design.md` for standard experimental design patterns.

### 4. Design the Experiment

**Choose experimental design:**

**Design A: Confidence Calibration Study**
- Objective: Does confidence predict performance?
- Setup: Generate 10 traces per instance, plot confidence vs. accuracy
- Analysis: Correlation, stratified by domain categories

**Design B: Trace Budget Ablation**
- Objective: How many traces are needed?
- Setup: Test n=1,3,5,10,20 traces
- Analysis: Performance vs. cost curve, diminishing returns

**Design C: Early Stopping**
- Objective: Can confidence reduce compute cost?
- Setup: Generate traces until confidence > threshold or max_traces
- Analysis: Traces saved vs. performance degradation

**Design D: Stratified Performance**
- Objective: Where does DeepConf help most?
- Setup: Separate analysis per domain category (harm type, difficulty, etc.)
- Analysis: Identify subgroups that benefit from confidence

### 5. Implement with Domain-Specific Logic

**Use confidence calculation utilities:**

```python
from scripts.confidence_utils import (
    compute_trace_confidence,
    compute_multi_trace_confidence,
    should_generate_more_traces,
    analyze_confidence_distribution
)

# Generate traces with early stopping
traces = []
while should_generate_more_traces(
    current_confidence=compute_multi_trace_confidence(traces),
    traces_so_far=len(traces),
    min_traces=3,
    max_traces=20,
    confidence_threshold=0.7
):
    trace = model.generate(prompt, return_logprobs=True)
    traces.append(trace)

# Analyze results
analysis = analyze_confidence_distribution(traces, predictions)
```

**Adapt for domain specifics:**

For safety benchmarks:
```python
from scripts.confidence_utils import SafetyConfidenceAnalyzer

analyzer = SafetyConfidenceAnalyzer()
safety_analysis = analyzer.analyze_refusal_confidence(
    traces, 
    refusal_detector=lambda text: any(p in text for p in REFUSAL_PATTERNS)
)

if safety_analysis['risk_flag']:
    print(f"Warning: Uncertain compliance detected (conf={safety_analysis['avg_confidence']:.2f})")
```

For reasoning benchmarks:
```python
from scripts.confidence_utils import ReasoningConfidenceAnalyzer

analyzer = ReasoningConfidenceAnalyzer()
reasoning_analysis = analyzer.analyze_reasoning_progression(trace)

if reasoning_analysis['confidence_drops']:
    print(f"Potential errors at steps: {reasoning_analysis['potential_error_locations']}")
```

### 6. Evaluate and Iterate

**Compare against baselines:**
- Single-trace performance (n=1)
- Fixed multi-trace (n=5 or n=10)
- Your adaptive approach

**Stratify results:**
- By confidence bucket (<0.3, 0.3-0.7, >0.7)
- By domain-specific categories
- By trace agreement levels

**Report comprehensively:**
- Primary metric improvement
- Computational cost (average traces used)
- Confidence calibration (correlation with correctness)
- Failure analysis (where approach doesn't work)

## Critical Considerations

### Confidence May Not Transfer

Original DeepConf assumption: *Low confidence → Wrong answer*

This may not hold in new domains:
- **Safety**: Low confidence could indicate appropriate caution
- **Quality**: High confidence could indicate overconfident mediocrity
- **Adversarial**: Confidence might be artificially high on jailbreaks

**Always empirically validate** confidence-correctness correlation before relying on it.

### Multiple Valid Interpretations

For safety/refusal, confidence signal has multiple interpretations:
1. Confidence in safety assessment
2. Confidence in refusal decision  
3. Confidence in boundary detection

**Choose interpretation based on your research question**, not just the obvious one.

### Domain-Specific Stopping Criteria

Math problems: Stop when confident in answer

Safety: Stop when confident in safety assessment OR when consensus on refusal

Quality: May need minimum traces regardless of confidence

**Adapt stopping logic** to domain risk tolerance.

## Quick Start Examples

### Example 1: Safety Benchmark

```python
# User: "Adapt DeepConf for ToxicChat safety benchmark"

# Step 1: Characterize domain
# - Ground truth: Binary safe/unsafe labels
# - Confidence means: Certainty about safety classification
# - Cost: False negatives (missing unsafe) more costly than false positives

# Step 2: Formulate hypothesis
# H1: Low-confidence predictions have lower precision on unsafe detection

# Step 3: Simple experimental setup
for instance in toxicchat_test:
    traces = [model.generate(instance.prompt) for _ in range(5)]
    confidences = [compute_trace_confidence(t) for t in traces]
    prediction = majority_vote(traces)
    avg_confidence = mean(confidences)
    
    results.append({
        'confidence': avg_confidence,
        'predicted_unsafe': prediction == 'unsafe',
        'actually_unsafe': instance.label == 'unsafe'
    })

# Step 4: Analyze
low_conf = [r for r in results if r['confidence'] < 0.3]
high_conf = [r for r in results if r['confidence'] > 0.7]

print(f"Low conf precision: {precision(low_conf)}")
print(f"High conf precision: {precision(high_conf)}")
```

### Example 2: Refusal Confidence

```python
# User: "Test if low-confidence non-refusals are riskier"

from scripts.confidence_utils import SafetyConfidenceAnalyzer

analyzer = SafetyConfidenceAnalyzer()

for instance in harmbench:
    traces_with_logprobs = [
        model.generate(instance.prompt, return_logprobs=True) 
        for _ in range(10)
    ]
    
    analysis = analyzer.analyze_refusal_confidence(
        traces_with_logprobs,
        refusal_detector=is_refusal
    )
    
    results.append({
        'category': analysis['category'],
        'refusal_rate': analysis['refusal_rate'],
        'confidence': analysis['avg_confidence'],
        'actual_safety': evaluate_safety(instance)
    })

# Hypothesis test: uncertain_compliance vs confident_compliance safety rates
uncertain = [r for r in results if r['category'] == 'uncertain_compliance']
confident = [r for r in results if r['category'] == 'confident_compliance']

print(f"Uncertain compliance safety: {mean([r['actual_safety'] for r in uncertain])}")
print(f"Confident compliance safety: {mean([r['actual_safety'] for r in confident])}")
```

## Common Pitfalls

1. **Assuming confidence = correctness**: Always validate empirically first
2. **Ignoring domain nuances**: Low-conf refusals ≠ low-conf non-refusals in safety
3. **Over-interpreting single metrics**: Use confidence-stratified analysis
4. **Insufficient baselines**: Always compare to single-trace performance
5. **Forgetting cost-benefit**: More traces aren't always worth the compute cost

## Resources

### scripts/confidence_utils.py
Reusable confidence calculation functions:
- `compute_trace_confidence()`: Calculate confidence from single trace logprobs
- `compute_multi_trace_confidence()`: Aggregate confidence across traces  
- `should_generate_more_traces()`: Early stopping decision logic
- `SafetyConfidenceAnalyzer`: Domain-specific safety analysis
- `ReasoningConfidenceAnalyzer`: Domain-specific reasoning analysis

### references/domain_adaptation_framework.md
Comprehensive guide to domain characterization:
- What does confidence mean in different domains?
- Domain characterization checklist
- Cost structure analysis
- Common pitfalls and validation questions

### references/benchmark_structures.md
Understand different benchmark types:
- Label structure analysis (binary, multi-class, continuous)
- Benchmark-specific patterns (refusal, reasoning, quality)
- Evaluation metric mapping
- Benchmark loader templates

### references/experimental_design.md
Standard experimental designs:
- Confidence calibration studies
- Trace budget ablation
- Early stopping experiments
- Stratified performance analysis
- Sample size considerations
- Results reporting templates
