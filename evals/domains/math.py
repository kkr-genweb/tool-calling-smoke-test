"""
Math Domain
==========

Mock math API for testing agent interactions with mathematical computations.
"""

import math
import statistics
from typing import List, Sequence

from pydantic_ai import RunContext

from ..base import BaseDomainClient, CallLogger


class MathAPIClient(BaseDomainClient):
    """Mock math API client."""
    
    def __init__(self, logger: CallLogger) -> None:
        super().__init__(logger)

    @property
    def domain_name(self) -> str:
        return "math"

    async def logarithm(self, x: float, base: float = math.e) -> float:
        """Calculate logarithm."""
        val = math.log(x, base)
        return self.logger.log("math", "logarithm", {"x": x, "base": base}, val)

    async def mean(self, data: Sequence[float]) -> float:
        """Calculate mean of a dataset."""
        val = statistics.fmean(data)
        return self.logger.log("math", "mean", {"data": list(data)}, val)

    async def standard_deviation(self, data: Sequence[float]) -> float:
        """Calculate standard deviation of a dataset."""
        # population stdev for simplicity
        m = statistics.fmean(data)
        var = sum((d - m) ** 2 for d in data) / len(data)
        val = math.sqrt(var)
        return self.logger.log("math", "standard_deviation", {"data": list(data)}, val)


def register_math_tools(agent, deps_type):
    """Register math tools with the agent."""
    
    @agent.tool
    async def math_log(ctx: RunContext[deps_type], x: float, base: float = math.e) -> float:
        """Calculate logarithm."""
        return await ctx.deps.math.logarithm(x, base)

    @agent.tool
    async def math_mean(ctx: RunContext[deps_type], data: List[float]) -> float:
        """Calculate mean."""
        return await ctx.deps.math.mean(data)

    @agent.tool
    async def math_std(ctx: RunContext[deps_type], data: List[float]) -> float:
        """Calculate standard deviation."""
        return await ctx.deps.math.standard_deviation(data)