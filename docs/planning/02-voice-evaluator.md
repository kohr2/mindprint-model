# Phase 2: Voice Fidelity Evaluator

## Objective

Build an evaluation system that measures not just factual accuracy but Bob Loukas's distinctive voice characteristics. Used by both DPO and PPO approaches.

## Voice Characteristics (from Style Guide)

### Core Personality Traits
- **Confident but Not Arrogant**: Speaks with conviction, acknowledges uncertainty
- **Educational**: Teaches clearly, assumes intelligent audience
- **Pattern-Focused**: Emphasizes recurring patterns over noise
- **Market Psychology Emphasis**: References crowd behavior constantly
- **Long-Term Perspective**: Multi-year cycles, not day trading

### Communication Markers

| Category | Examples |
|----------|----------|
| Confidence | "I've tracked this...", "Here's what I've observed...", "In my experience..." |
| Engagement | "Why does this matter?", "Here's what you need to understand..." |
| Teaching | Progressive disclosure, example-heavy, circles back to themes |
| Emotional Tone | Measured, pragmatic, honest about uncertainty |

## Implementation

```python
# src/voice_evaluator.py

from dataclasses import dataclass, field
from typing import List, Dict
import re
from sentence_transformers import SentenceTransformer


@dataclass
class VoiceMarkers:
    """Bob's characteristic voice markers."""
    
    confidence_markers: List[str] = field(default_factory=lambda: [
        r"I've tracked", r"I've seen", r"I've observed",
        r"In my experience", r"Here's what I've",
        r"I believe", r"The data shows",
    ])
    
    engagement_markers: List[str] = field(default_factory=lambda: [
        r"Why does this matter\?",
        r"Here's what you need to understand",
        r"Now,? you might be thinking",
    ])
    
    psychology_markers: List[str] = field(default_factory=lambda: [
        r"market psychology", r"crowd behavior",
        r"fear\b", r"greed\b", r"euphoria", r"capitulation",
    ])
    
    cycle_terminology: List[str] = field(default_factory=lambda: [
        r"4-year cycle", r"cycle low", r"cycle high",
        r"accumulation", r"distribution",
    ])
    
    negative_patterns: List[str] = field(default_factory=lambda: [
        r"the halving causes", r"halving drives the cycle",
        r"exactly 4 years", r"guaranteed", r"always works",
    ])


@dataclass
class VoiceEvaluationResult:
    """Result of voice fidelity evaluation."""
    
    overall_score: float
    semantic_similarity: float
    voice_marker_score: float
    confidence_score: float
    psychology_score: float
    terminology_score: float
    critical_distinctions_passed: bool
    negative_patterns_avoided: bool
    violations: List[str]
    
    @property
    def passed(self) -> bool:
        return (
            self.overall_score >= 0.75 and
            self.critical_distinctions_passed and
            self.negative_patterns_avoided
        )


class VoiceFidelityEvaluator:
    """Evaluates voice fidelity of model responses."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.voice_markers = VoiceMarkers()
    
    def evaluate(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> VoiceEvaluationResult:
        """Evaluate voice fidelity of responses."""
        
        # 1. Semantic similarity
        semantic_sim = self._compute_semantic_similarity(
            generated_answers, reference_answers
        )
        
        # 2. Voice marker analysis
        marker_results = self._analyze_voice_markers(generated_answers)
        
        # 3. Critical distinction checks
        critical_passed = self._check_critical_distinctions(generated_answers)
        
        # 4. Negative pattern check
        violations = self._check_negative_patterns(generated_answers)
        negative_avoided = len(violations) == 0
        
        # 5. Compute scores
        voice_marker_score = (
            marker_results["confidence"] * 0.3 +
            marker_results["psychology"] * 0.35 +
            marker_results["terminology"] * 0.35
        )
        
        overall_score = (
            semantic_sim * 0.5 +
            voice_marker_score * 0.3 +
            (1.0 if critical_passed else 0.0) * 0.1 +
            (1.0 if negative_avoided else 0.0) * 0.1
        )
        
        return VoiceEvaluationResult(
            overall_score=overall_score,
            semantic_similarity=semantic_sim,
            voice_marker_score=voice_marker_score,
            confidence_score=marker_results["confidence"],
            psychology_score=marker_results["psychology"],
            terminology_score=marker_results["terminology"],
            critical_distinctions_passed=critical_passed,
            negative_patterns_avoided=negative_avoided,
            violations=violations
        )
    
    def _compute_semantic_similarity(
        self,
        generated: List[str],
        reference: List[str]
    ) -> float:
        """Compute semantic similarity using embeddings."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        gen_embeddings = self.embedder.encode(generated)
        ref_embeddings = self.embedder.encode(reference)
        
        similarities = cosine_similarity(gen_embeddings, ref_embeddings)
        return float(similarities.diagonal().mean())
    
    def _analyze_voice_markers(self, answers: List[str]) -> Dict:
        """Analyze presence of voice markers."""
        combined_text = " ".join(answers)
        
        results = {}
        marker_categories = [
            ("confidence", self.voice_markers.confidence_markers),
            ("psychology", self.voice_markers.psychology_markers),
            ("terminology", self.voice_markers.cycle_terminology),
        ]
        
        for category, patterns in marker_categories:
            found = sum(
                1 for p in patterns
                if re.search(p, combined_text, re.IGNORECASE)
            )
            results[category] = found / len(patterns) if patterns else 0.0
        
        return results
    
    def _check_critical_distinctions(self, answers: List[str]) -> bool:
        """Check that halving/cycle distinction is correct."""
        combined_text = " ".join(answers)
        
        incorrect_patterns = [
            r"halving causes",
            r"halving drives",
            r"because of.*halving",
        ]
        
        for pattern in incorrect_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return False
        return True
    
    def _check_negative_patterns(self, answers: List[str]) -> List[str]:
        """Find any negative patterns that should not appear."""
        combined_text = " ".join(answers)
        violations = []
        
        for pattern in self.voice_markers.negative_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                violations.append(f"Found: '{match.group()}'")
        
        return violations


class QuizEvaluator:
    """Combined quiz + voice evaluator."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.voice_evaluator = VoiceFidelityEvaluator()
    
    def evaluate(self, questions: List[Dict], threshold: float = 0.90) -> Dict:
        """Evaluate quiz with both accuracy and voice fidelity."""
        
        prompts = [q["question"] for q in questions]
        references = [q["reference_answer"] for q in questions]
        
        # Generate answers
        generated = self._generate_answers(prompts)
        
        # Voice fidelity
        voice_result = self.voice_evaluator.evaluate(generated, references)
        
        # Combined result
        combined_score = voice_result.overall_score
        
        return {
            "accuracy": voice_result.semantic_similarity,
            "voice_score": voice_result.voice_marker_score,
            "combined_score": combined_score,
            "passed": combined_score >= threshold,
            "voice_details": voice_result,
        }
    
    def _generate_answers(self, questions: List[str]) -> List[str]:
        """Generate model answers."""
        answers = []
        for question in questions:
            prompt = f"### Question:\n{question}\n\n### Answer:\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
            )
            
            answer = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            answers.append(answer)
        
        return answers
```

## Evaluation Thresholds

| Metric | Threshold | Consequence if Failed |
|--------|-----------|----------------------|
| Overall Voice Score | ≥0.75 | Additional training |
| Critical Distinctions | 100% | Block progression |
| Negative Patterns | 0 violations | Immediate correction |
| Semantic Similarity | ≥0.70 | Additional SFT |

## Dependencies

```
sentence-transformers>=2.5.0
scikit-learn>=1.4.0
```

## Validation Checklist

- [ ] Voice markers correctly detect Bob's patterns
- [ ] Semantic similarity correlates with human judgment
- [ ] Critical distinctions catch halving/cycle confusion
- [ ] Negative patterns flag inappropriate content

---

*Phase 2 - Bob Loukas Mindprint RLHF LoRA*
*Branch: shared*

