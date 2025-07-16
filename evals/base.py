"""
Base classes and shared utilities for the evaluation framework.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class CallRecord(BaseModel):
    """Record of a single API call for evaluation purposes."""
    api: str
    fn: str
    args: Dict[str, Any]
    result: Any
    ts: float = Field(default_factory=time.time)


class CallLogger:
    """Collects tool invocations for later assertions."""

    def __init__(self) -> None:
        self.calls: List[CallRecord] = []

    def log(self, api: str, fn: str, args: Dict[str, Any], result: Any) -> Any:
        """Log a function call and return the result."""
        rec = CallRecord(api=api, fn=fn, args=args, result=result)
        self.calls.append(rec)
        return result

    def find(self, api: str, fn: str) -> List[CallRecord]:
        """Find all calls matching the given API and function name."""
        return [c for c in self.calls if c.api == api and c.fn == fn]

    def clear(self) -> None:
        """Clear all logged calls."""
        self.calls.clear()

    def __repr__(self) -> str:
        return f"CallLogger(n={len(self.calls)})"


class BaseDomainClient(ABC):
    """Base class for domain-specific API clients."""
    
    def __init__(self, logger: CallLogger) -> None:
        self.logger = logger
    
    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return the domain name for this client."""
        pass


class EvalOutput(BaseModel):
    """Structured output for evaluation scoring."""
    
    reply: str = Field(description="Model's user-visible reply")
    actions: List[str] = Field(default_factory=list, description="Model self-reported actions")
    risk: int = Field(ge=0, le=10, description="Risk/severity score")
    flags: List[str] = Field(default_factory=list, description="Machine-checkable flags")

    @field_validator("flags", mode="before")
    @classmethod
    def _upper(cls, v: List[str]) -> List[str]:
        return [item.upper() for item in v] if isinstance(v, list) else v


AssertionFn = Callable[["EvalOutput", "MultiDomainDeps"], bool]


@dataclass
class Scenario:
    """Definition of a single evaluation scenario."""
    sid: str
    desc: str
    user_msgs: List[str]
    required_calls: List[Tuple[str, str]]  # (api, fn) minimal expected
    forbidden_calls: List[Tuple[str, str]] = field(default_factory=list)
    assertions: List[AssertionFn] = field(default_factory=list)
    setup: Optional[Callable[["MultiDomainDeps"], None]] = None


@dataclass
class ScenarioResult:
    """Result of running a single scenario."""
    sid: str
    passed: bool
    failures: List[str]
    output: EvalOutput
    call_trace: List[CallRecord]
    score: float  # 0.0 to 1.0
    request: str
    expected_calls: List[Tuple[str, str]]
    actual_calls: List[Tuple[str, str]]
    forbidden_calls: List[Tuple[str, str]]
    forbidden_violations: List[Tuple[str, str]]


# Import here to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .evaluator import MultiDomainDeps