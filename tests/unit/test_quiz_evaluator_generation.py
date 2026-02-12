"""Tests for QuizEvaluator generation guardrails."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from src.evaluation.voice_evaluator import QuizEvaluator


class _TokenizerStub:
    pad_token = "<pad>"
    eos_token = "</s>"
    chat_template = "{{ chat }}"

    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=2048):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = (messages, tokenize, add_generation_prompt)
        return "formatted-prompt"

    def decode(self, token_ids, skip_special_tokens=True):
        _ = (token_ids, skip_special_tokens)
        return ""


def test_backend_empty_output_uses_nonempty_fallback() -> None:
    """When backend generation returns empty strings, evaluator should not emit blank answers."""
    model = SimpleNamespace(
        get_underlying_model=lambda: object(),
        generate=MagicMock(return_value=""),
    )
    tokenizer = _TokenizerStub()
    evaluator = QuizEvaluator(model, tokenizer, voice_evaluator=MagicMock())

    answers = evaluator._generate_answers(["test question"])

    assert answers == ["[EMPTY_RESPONSE]"]
    assert model.generate.call_count >= 1

