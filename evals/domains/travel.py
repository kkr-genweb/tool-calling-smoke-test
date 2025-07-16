"""
Travel Booking Domain
=====================

Mock travel booking API for testing agent interactions with travel services.
"""

from typing import Any, Dict, Optional, Tuple

from pydantic_ai import RunContext

from ..base import BaseDomainClient, CallLogger


class TravelBookingClient(BaseDomainClient):
    """Mock travel booking API client."""
    
    def __init__(self, logger: CallLogger, *, airport_by_city: Optional[Dict[str, str]] = None, 
                 base_fares: Optional[Dict[Tuple[str, str], float]] = None, insurance_price: float = 42.0) -> None:
        super().__init__(logger)
        self.airport_by_city = airport_by_city or {
            "Austin": "AUS", "Round Rock": "AUS", "Denver": "DEN", 
            "Chicago": "ORD", "Toronto": "YYZ"
        }
        self.base_fares = base_fares or {("DEN", "ORD"): 189.0, ("AUS", "YYZ"): 580.0}
        self.insurance_price = insurance_price

    @property
    def domain_name(self) -> str:
        return "travel"

    async def get_nearest_airport_by_city(self, city: str) -> str:
        """Get nearest airport code for a city."""
        code = self.airport_by_city.get(city, "UNKNOWN")
        return self.logger.log("travel", "get_nearest_airport_by_city", {"city": city}, code)

    async def book_flight(self, origin: str, dest: str, passengers: int = 1, 
                         max_price: Optional[float] = None) -> Dict[str, Any]:
        """Book a flight between two airports."""
        base = self.base_fares.get((origin, dest), 9999.0)
        total = base * passengers
        booked = max_price is None or total <= max_price
        result = {
            "origin": origin, "dest": dest, "passengers": passengers, 
            "base_fare": base, "total": total, "booked": booked
        }
        return self.logger.log("travel", "book_flight", 
                              {"origin": origin, "dest": dest, "passengers": passengers, "max_price": max_price}, 
                              result)

    async def purchase_insurance(self, booking_ref: str, passengers: int = 1) -> Dict[str, Any]:
        """Purchase travel insurance for a booking."""
        total = self.insurance_price * passengers
        result = {"booking_ref": booking_ref, "passengers": passengers, "insurance_total": total, "purchased": True}
        return self.logger.log("travel", "purchase_insurance", 
                              {"booking_ref": booking_ref, "passengers": passengers}, result)


def register_travel_tools(agent, deps_type):
    """Register travel booking tools with the agent."""
    
    @agent.tool
    async def travel_get_airport(ctx: RunContext[deps_type], city: str) -> str:
        """Get nearest airport for a city."""
        return await ctx.deps.travel.get_nearest_airport_by_city(city)

    @agent.tool
    async def travel_book_flight(ctx: RunContext[deps_type], origin: str, dest: str, 
                                passengers: int = 1, max_price: float | None = None) -> Dict[str, Any]:
        """Book a flight."""
        return await ctx.deps.travel.book_flight(origin, dest, passengers, max_price)

    @agent.tool
    async def travel_purchase_insurance(ctx: RunContext[deps_type], booking_ref: str, 
                                       passengers: int = 1) -> Dict[str, Any]:
        """Purchase travel insurance."""
        return await ctx.deps.travel.purchase_insurance(booking_ref, passengers)