"""
Vehicle Control Domain
=====================

Mock vehicle control API for testing agent interactions with vehicle systems.
"""

from typing import Any, Dict

from pydantic_ai import RunContext

from ..base import BaseDomainClient, CallLogger


class VehicleControlClient(BaseDomainClient):
    """Mock vehicle control API client."""
    
    def __init__(self, logger: CallLogger, *, engine_initially_on: bool = False, battery_pct: float = 87.0) -> None:
        super().__init__(logger)
        self.engine_on = engine_initially_on
        self.battery_pct = battery_pct

    @property
    def domain_name(self) -> str:
        return "vehicle"

    async def startEngine(self) -> str:
        """Start the vehicle engine."""
        self.engine_on = True
        return self.logger.log("vehicle", "startEngine", {}, "engine_started")

    async def displayCarStatus(self) -> Dict[str, Any]:
        """Get current vehicle status."""
        status = {"engine_on": self.engine_on, "battery_pct": self.battery_pct}
        return self.logger.log("vehicle", "displayCarStatus", {}, status)

    async def estimate_distance(self, miles: float) -> Any:
        """Estimate if vehicle can travel the requested distance."""
        if not self.engine_on:
            result = {"error": "engine_off"}
        else:
            # pretend range = battery_pct * 3 miles per %
            result = min(miles, self.battery_pct * 3.0)
        return self.logger.log("vehicle", "estimate_distance", {"miles": miles}, result)


def register_vehicle_tools(agent, deps_type):
    """Register vehicle control tools with the agent."""
    
    @agent.tool
    async def start_engine(ctx: RunContext[deps_type]) -> str:
        """Start the vehicle engine."""
        return await ctx.deps.vehicle.startEngine()

    @agent.tool
    async def vehicle_status(ctx: RunContext[deps_type]) -> Dict[str, Any]:
        """Return vehicle status (engine & battery%)."""
        return await ctx.deps.vehicle.displayCarStatus()

    @agent.tool
    async def vehicle_estimate_distance(ctx: RunContext[deps_type], miles: float) -> Any:
        """Estimate if the vehicle can travel the requested miles (requires engine on)."""
        return await ctx.deps.vehicle.estimate_distance(miles)