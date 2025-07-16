"""
Domain-specific API clients for evaluation scenarios.
"""

from .vehicle import VehicleControlClient, register_vehicle_tools
from .trading import TradingBotClient, register_trading_tools
from .travel import TravelBookingClient, register_travel_tools
from .filesystem import GorillaFSClient, register_filesystem_tools
from .messaging import MessageAPIClient, register_messaging_tools
from .social import TwitterAPIClient, register_social_tools
from .ticketing import TicketAPIClient, register_ticketing_tools
from .math import MathAPIClient, register_math_tools

__all__ = [
    "VehicleControlClient",
    "TradingBotClient", 
    "TravelBookingClient",
    "GorillaFSClient",
    "MessageAPIClient",
    "TwitterAPIClient",
    "TicketAPIClient",
    "MathAPIClient",
    "register_vehicle_tools",
    "register_trading_tools",
    "register_travel_tools", 
    "register_filesystem_tools",
    "register_messaging_tools",
    "register_social_tools",
    "register_ticketing_tools",
    "register_math_tools",
]