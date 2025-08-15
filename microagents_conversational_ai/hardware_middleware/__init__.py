"""
Hardware Middleware Package
Provides device abstraction, communication bus, and telemetry for AI orchestrator.
"""

from .device_manager import DeviceManager, Device
from .message_bus import MessageBus, Message
from .discovery_service import DeviceDiscoveryService, DeviceEvent
from .telemetry_aggregator import TelemetryAggregator, TelemetryReading
from .command_validator import CommandValidator, CommandValidationRequest, CommandValidationResponse

__all__ = [
    'DeviceManager',
    'Device',
    'MessageBus',
    'Message', 
    'DeviceDiscoveryService',
    'DeviceEvent',
    'TelemetryAggregator',
    'TelemetryReading',
    'CommandValidator',
    'CommandValidationRequest',
    'CommandValidationResponse'
]
