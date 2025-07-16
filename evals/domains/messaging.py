"""
Messaging Domain
===============

Mock messaging API for testing agent interactions with messaging services.
"""

from typing import Any, Dict, List

from pydantic_ai import RunContext

from ..base import BaseDomainClient, CallLogger


class MessageAPIClient(BaseDomainClient):
    """Mock messaging API client."""
    
    def __init__(self, logger: CallLogger) -> None:
        super().__init__(logger)
        self.inbox: Dict[str, str] = {}
        self.sent: Dict[str, str] = {}
        self._id = 0

    @property
    def domain_name(self) -> str:
        return "msg"

    def _new_id(self) -> str:
        """Generate a new message ID."""
        self._id += 1
        return f"m{self._id}"

    async def send_message(self, to: str, body: str) -> Dict[str, Any]:
        """Send a message to a recipient."""
        mid = self._new_id()
        self.sent[mid] = body
        self.inbox.setdefault(to, "")
        return self.logger.log("msg", "send_message", {"to": to, "body": body}, {"id": mid, "sent": True})

    async def delete_message(self, msg_id: str) -> bool:
        """Delete a message by ID."""
        existed = msg_id in self.sent
        if existed:
            del self.sent[msg_id]
        return self.logger.log("msg", "delete_message", {"msg_id": msg_id}, existed)

    async def view_messages_received(self, who: str) -> List[str]:
        """View messages received by a user."""
        # Minimal stub implementation
        msgs = [v for k, v in self.sent.items() if who in k]  # silly filter
        return self.logger.log("msg", "view_messages_received", {"who": who}, msgs)


def register_messaging_tools(agent, deps_type):
    """Register messaging tools with the agent."""
    
    @agent.tool
    async def msg_send(ctx: RunContext[deps_type], to: str, body: str) -> Dict[str, Any]:
        """Send a message."""
        return await ctx.deps.msg.send_message(to, body)

    @agent.tool
    async def msg_delete(ctx: RunContext[deps_type], msg_id: str) -> bool:
        """Delete a message."""
        return await ctx.deps.msg.delete_message(msg_id)

    @agent.tool
    async def msg_view(ctx: RunContext[deps_type], who: str) -> List[str]:
        """View received messages."""
        return await ctx.deps.msg.view_messages_received(who)