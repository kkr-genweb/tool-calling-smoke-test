"""
Trading Bot Domain
==================

Mock trading API for testing agent interactions with financial markets.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic_ai import RunContext

from ..base import BaseDomainClient, CallLogger


class TradingBotClient(BaseDomainClient):
    """Mock trading bot API client."""
    
    def __init__(self, logger: CallLogger, *, prices: Optional[Dict[str, float]] = None, 
                 cash: float = 10000.0, watchlist: Optional[List[str]] = None) -> None:
        super().__init__(logger)
        self.prices = prices or {"AAPL": 225.50, "TSLA": 245.10, "SPY": 560.0}
        self.cash = cash
        self.watchlist = watchlist or ["AAPL", "TSLA", "NVDA"]
        self.orders: List[Dict[str, Any]] = []

    @property
    def domain_name(self) -> str:
        return "trading"

    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get stock information by symbol."""
        price = self.prices.get(symbol.upper())
        if price is None:
            result = {"symbol": symbol.upper(), "error": "not_found"}
        else:
            result = {"symbol": symbol.upper(), "price": price}
        return self.logger.log("trading", "get_stock_info", {"symbol": symbol}, result)

    async def place_order(self, symbol: str, qty: int, side: Literal["buy", "sell"] = "buy") -> Dict[str, Any]:
        """Place a trading order."""
        cost = self.prices.get(symbol.upper(), 0.0) * qty
        status = "filled" if side == "sell" or cost <= self.cash else "rejected_insufficient_funds"
        if status == "filled" and side == "buy":
            self.cash -= cost
        if status == "filled" and side == "sell":
            self.cash += cost
        order = {"symbol": symbol.upper(), "qty": qty, "side": side, "status": status, "cost": cost}
        self.orders.append(order)
        return self.logger.log("trading", "place_order", order, order)

    async def get_watchlist(self) -> List[str]:
        """Get the user's watchlist."""
        return self.logger.log("trading", "get_watchlist", {}, list(self.watchlist))


def register_trading_tools(agent, deps_type):
    """Register trading tools with the agent."""
    
    @agent.tool
    async def trading_get_stock(ctx: RunContext[deps_type], symbol: str) -> Dict[str, Any]:
        """Get stock information."""
        return await ctx.deps.trading.get_stock_info(symbol)

    @agent.tool
    async def trading_place_order(ctx: RunContext[deps_type], symbol: str, qty: int, side: str = "buy") -> Dict[str, Any]:
        """Place a trading order."""
        return await ctx.deps.trading.place_order(symbol, qty, side)  # type: ignore[arg-type]

    @agent.tool
    async def trading_get_watchlist(ctx: RunContext[deps_type]) -> List[str]:
        """Get user's watchlist."""
        return await ctx.deps.trading.get_watchlist()