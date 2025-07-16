"""
Ticketing Domain
===============

Mock ticketing API for testing agent interactions with support systems.
"""

from typing import Any, Dict

from pydantic_ai import RunContext

from ..base import BaseDomainClient, CallLogger


class TicketAPIClient(BaseDomainClient):
    """Mock ticketing API client."""
    
    def __init__(self, logger: CallLogger) -> None:
        super().__init__(logger)
        self.tickets: Dict[str, Dict[str, Any]] = {}
        self._id = 0

    @property
    def domain_name(self) -> str:
        return "ticket"

    def _new_id(self) -> str:
        """Generate a new ticket ID."""
        self._id += 1
        return f"t{self._id}"

    async def create_ticket(self, title: str, body: str, severity: int = 3) -> Dict[str, Any]:
        """Create a new support ticket."""
        tid = self._new_id()
        ticket = {"id": tid, "title": title, "body": body, "severity": severity, "status": "open"}
        self.tickets[tid] = ticket
        return self.logger.log("ticket", "create_ticket", {"title": title, "body": body, "severity": severity}, ticket)

    async def get_ticket(self, ticket_id: str) -> Dict[str, Any]:
        """Get a ticket by ID."""
        ticket = self.tickets.get(ticket_id, {"error": "not_found"})
        return self.logger.log("ticket", "get_ticket", {"ticket_id": ticket_id}, ticket)

    async def close_ticket(self, ticket_id: str) -> bool:
        """Close a ticket by ID."""
        t = self.tickets.get(ticket_id)
        if t:
            t["status"] = "closed"
            result = True
        else:
            result = False
        return self.logger.log("ticket", "close_ticket", {"ticket_id": ticket_id}, result)


def register_ticketing_tools(agent, deps_type):
    """Register ticketing tools with the agent."""
    
    @agent.tool
    async def ticket_create(ctx: RunContext[deps_type], title: str, body: str, severity: int = 3) -> Dict[str, Any]:
        """Create a support ticket."""
        return await ctx.deps.ticket.create_ticket(title, body, severity)

    @agent.tool
    async def ticket_get(ctx: RunContext[deps_type], ticket_id: str) -> Dict[str, Any]:
        """Get a ticket by ID."""
        return await ctx.deps.ticket.get_ticket(ticket_id)

    @agent.tool
    async def ticket_close(ctx: RunContext[deps_type], ticket_id: str) -> bool:
        """Close a ticket."""
        return await ctx.deps.ticket.close_ticket(ticket_id)