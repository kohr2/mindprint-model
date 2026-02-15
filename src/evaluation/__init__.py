"""Evaluation module for Bob Loukas mindprint models."""

from typing import Any

__all__ = []


def _export(name: str, value: Any) -> None:
    globals()[name] = value
    __all__.append(name)


from .voice_markers import VoiceMarkers

_export("VoiceMarkers", VoiceMarkers)

try:
    from .voice_evaluator import VoiceFidelityEvaluator, VoiceEvaluationResult

    _export("VoiceFidelityEvaluator", VoiceFidelityEvaluator)
    _export("VoiceEvaluationResult", VoiceEvaluationResult)
except Exception:
    # Allow importing lightweight utilities (e.g. voice markers) without
    # requiring full evaluation dependencies in data-prep-only environments.
    pass

try:
    from .pipeline import (
        EvaluationPipeline,
        EvalLevel,
        QuestionResult,
        LevelResult,
        EvaluationReport,
    )
    from .reporting import ReportGenerator

    _export("EvaluationPipeline", EvaluationPipeline)
    _export("EvalLevel", EvalLevel)
    _export("QuestionResult", QuestionResult)
    _export("LevelResult", LevelResult)
    _export("EvaluationReport", EvaluationReport)
    _export("ReportGenerator", ReportGenerator)
except Exception:
    pass
