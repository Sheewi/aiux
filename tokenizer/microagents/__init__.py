"""
Microagents Specialization Matrix

A collection of specialized microagents for different domains.
"""

__version__ = "1.0.0"
__author__ = "Microagents Team"

from .web_automation import WebAutomationAgent
from .data_extraction import DataExtractionAgent
from .computer_vision import ComputerVisionAgent
from .system_governance import SystemGovernanceAgent
from .api_orchestration import APIOrchestrationAgent

__all__ = [
    "WebAutomationAgent",
    "DataExtractionAgent", 
    "ComputerVisionAgent",
    "SystemGovernanceAgent",
    "APIOrchestrationAgent"
]
