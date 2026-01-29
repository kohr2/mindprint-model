#!/usr/bin/env python3
"""
Compare performance of models trained on different datasets.

Evaluates models trained on:
- Dataset A: Transcripts only
- Dataset B: Combined (textbook + transcripts)
- Dataset C: Textbook only (baseline)

Generates comparison report with metrics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("ERROR: sentence-transformers and scikit-learn required. Run: pip install sentence-transformers scikit-learn")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare models trained on different datasets."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the comparator.

        Args:
            model_name: Sentence transformer model for semantic similarity
        """
        logger.info(f"Loading sentence transformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)

    def evaluate_knowledge_accuracy(
        self,
        model_responses: List[str],
        reference_answers: List[str],
    ) -> float:
        """
        Evaluate knowledge accuracy using semantic similarity.

        Args:
            model_responses: Model-generated answers
            reference_answers: Reference answers

        Returns:
            Average cosine similarity score
        """
        if not model_responses or not reference_answers:
            return 0.0

        # Encode responses
        response_embeddings = self.encoder.encode(model_responses)
        reference_embeddings = self.encoder.encode(reference_answers)

        # Calculate similarities
        similarities = cosine_similarity(response_embeddings, reference_embeddings)
        # Take diagonal (matching pairs)
        scores = np.diag(similarities)

        return float(np.mean(scores))

    def evaluate_voice_fidelity(
        self,
        model_responses: List[str],
        bob_reference: List[str],
    ) -> float:
        """
        Evaluate voice fidelity using semantic similarity to Bob's style.

        Args:
            model_responses: Model-generated answers
            bob_reference: Bob's reference answers

        Returns:
            Average similarity score
        """
        return self.evaluate_knowledge_accuracy(model_responses, bob_reference)

    def count_terminology_usage(self, text: str) -> Dict[str, int]:
        """
        Count usage of Bob-specific terminology.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with term counts
        """
        terms = {
            "DCL": 0,
            "right-translated": 0,
            "left-translated": 0,
            "Bressert": 0,
            "4-year cycle": 0,
            "40-week low": 0,
            "cycle low": 0,
            "accumulation": 0,
            "distribution": 0,
        }

        text_lower = text.lower()
        for term in terms.keys():
            terms[term] = text_lower.count(term.lower())

        return terms

    def evaluate_terminology(
        self, model_responses: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate terminology usage across responses.

        Args:
            model_responses: Model-generated answers

        Returns:
            Dictionary with average term usage rates
        """
        all_counts = {term: [] for term in [
            "DCL", "right-translated", "left-translated", "Bressert",
            "4-year cycle", "40-week low", "cycle low", "accumulation", "distribution"
        ]}

        for response in model_responses:
            counts = self.count_terminology_usage(response)
            for term, count in counts.items():
                all_counts[term].append(count)

        # Calculate averages
        averages = {}
        for term, counts in all_counts.items():
            total_terms = sum(counts)
            total_responses = len(counts)
            averages[term] = total_terms / total_responses if total_responses > 0 else 0.0

        return averages

    def compare_models(
        self,
        transcripts_model_path: Optional[str],
        combined_model_path: Optional[str],
        textbook_model_path: Optional[str],
        test_questions: List[Dict],
    ) -> Dict:
        """
        Compare all three models.

        Args:
            transcripts_model_path: Path to transcripts-only model
            combined_model_path: Path to combined model
            textbook_model_path: Path to textbook-only model
            test_questions: List of test questions with reference answers

        Returns:
            Comparison results dictionary
        """
        results = {
            "transcripts": None,
            "combined": None,
            "textbook": None,
        }

        # Note: Actual model inference would go here
        # For now, this is a placeholder structure
        logger.warning("Model inference not implemented. This is a placeholder structure.")
        logger.warning("You would need to load models and generate responses here.")

        return results

    def generate_report(
        self,
        results: Dict,
        output_path: Path,
    ):
        """
        Generate comparison report.

        Args:
            results: Comparison results
            output_path: Path to save report
        """
        report_lines = [
            "# Model Comparison Report",
            "",
            "## Datasets Compared",
            "",
            "- **Dataset A**: Transcripts only",
            "- **Dataset B**: Combined (textbook + transcripts)",
            "- **Dataset C**: Textbook only (baseline)",
            "",
            "## Metrics",
            "",
            "### Knowledge Accuracy",
            "Semantic similarity to reference answers (0-1 scale)",
            "",
            "### Voice Fidelity",
            "Semantic similarity to Bob's style (0-1 scale)",
            "",
            "### Terminology Usage",
            "Average usage of Bob-specific terms per response",
            "",
            "## Results",
            "",
            "```json",
            json.dumps(results, indent=2),
            "```",
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"Report saved to {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare models trained on different datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--transcripts-model",
        help="Path to transcripts-only model checkpoint",
    )
    parser.add_argument(
        "--combined-model",
        help="Path to combined model checkpoint",
    )
    parser.add_argument(
        "--textbook-model",
        help="Path to textbook-only model checkpoint",
    )
    parser.add_argument(
        "--test-questions",
        help="Path to test questions JSON file",
    )
    parser.add_argument(
        "--output",
        default="./reports/model_comparison.md",
        help="Output path for comparison report",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load test questions if provided
    test_questions = []
    if args.test_questions:
        with open(args.test_questions, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    # Initialize comparator
    comparator = ModelComparator()

    # Compare models
    results = comparator.compare_models(
        transcripts_model_path=args.transcripts_model,
        combined_model_path=args.combined_model,
        textbook_model_path=args.textbook_model,
        test_questions=test_questions,
    )

    # Generate report
    output_path = Path(args.output)
    comparator.generate_report(results, output_path)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Report saved to: {output_path.absolute()}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
