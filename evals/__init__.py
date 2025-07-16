"""
Multi-Domain PydanticAI Evaluation Suite
========================================

A modular evaluation framework for testing AI agents across multiple domains.
"""

from .base import (
    CallRecord,
    CallLogger,
    BaseDomainClient,
    EvalOutput,
    Scenario,
    ScenarioResult,
    AssertionFn,
)

from .evaluator import MultiDomainEvaluator

__all__ = [
    "CallRecord",
    "CallLogger", 
    "BaseDomainClient",
    "EvalOutput",
    "Scenario",
    "ScenarioResult",
    "AssertionFn",
    "MultiDomainEvaluator",
]