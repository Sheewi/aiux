"""
Agent Registry System
Dynamic agent discovery and management system for the microagent ecosystem.
"""

import os
import importlib
import inspect
from typing import Dict, List, Type, Optional
from pathlib import Path
from core.base_agent import MicroAgent, HybridAgent

class AgentRegistry:
    """
    Central registry for discovering and managing agents.
    Singleton pattern ensures consistent agent discovery across the system.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
            cls._instance._hybrid_agents = {}
            cls._instance._agent_metadata = {}  # New: metadata for each agent
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._discover_agents()
            self._initialized = True
    
    def _discover_agents(self):
        """Automatically discover all available agents."""
        current_dir = Path(__file__).parent
        agents_dir = current_dir / "agents"
        hybrids_dir = current_dir / "hybrids"
        
        # Discover individual agents
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.py"):
                if agent_file.name.startswith("__"):
                    continue
                    
                try:
                    module_name = f"agents.{agent_file.stem}"
                    module = importlib.import_module(module_name)
                    
                    # Find agent classes in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, MicroAgent) and 
                            obj != MicroAgent and 
                            obj != HybridAgent):
                            self._agents[name] = obj
                            
                except ImportError as e:
                    print(f"Failed to import {agent_file}: {e}")
        
        # Discover hybrid agents
        if hybrids_dir.exists():
            for hybrid_file in hybrids_dir.glob("hybrid_*.py"):
                try:
                    module_name = f"hybrids.{hybrid_file.stem}"
                    module = importlib.import_module(module_name)
                    
                    # Find hybrid agent classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, HybridAgent) and 
                            obj != HybridAgent):
                            self._hybrid_agents[name] = obj
                            
                except ImportError as e:
                    print(f"Failed to import {hybrid_file}: {e}")
    
    def register(self, agent_class: Type[MicroAgent], force: bool = False, metadata: Optional[dict] = None):
        """
        Manually register an agent class and its metadata.
        Args:
            agent_class: The agent class to register
            force: Whether to overwrite existing registration
            metadata: Dict with keys: 'capabilities', 'token_formats', 'resource_footprint'
        """
        class_name = agent_class.__name__
        if class_name in self._agents and not force:
            raise ValueError(f"Agent {class_name} already registered")
        if issubclass(agent_class, HybridAgent):
            self._hybrid_agents[class_name] = agent_class
        else:
            self._agents[class_name] = agent_class
        # Store metadata
        if metadata:
            self._agent_metadata[class_name] = metadata
        else:
            # Try to get from class attributes
            self._agent_metadata[class_name] = {
                'capabilities': getattr(agent_class, 'capabilities', []),
                'token_formats': getattr(agent_class, 'token_formats', []),
                'resource_footprint': getattr(agent_class, 'resource_footprint', {})
            }
        return agent_class
    def get_agent_metadata(self, name: str) -> Optional[dict]:
        """Get metadata for an agent by name."""
        return self._agent_metadata.get(name)

    def list_agents_with_metadata(self, include_hybrids: bool = True) -> dict:
        """List all agents and their metadata."""
        agents = self.list_agents(include_hybrids=include_hybrids)
        return {name: self._agent_metadata.get(name, {}) for name in agents}
    
    def get_agent(self, name: str) -> Optional[Type[MicroAgent]]:
        """Get an agent class by name."""
        return self._agents.get(name) or self._hybrid_agents.get(name)
    
    def list_agents(self, include_hybrids: bool = True) -> Dict[str, Type[MicroAgent]]:
        """List all registered agents."""
        agents = self._agents.copy()
        if include_hybrids:
            agents.update(self._hybrid_agents)
        return agents
    
    def list_individual_agents(self) -> Dict[str, Type[MicroAgent]]:
        """List only individual (non-hybrid) agents."""
        return self._agents.copy()
    
    def list_hybrid_agents(self) -> Dict[str, Type[HybridAgent]]:
        """List only hybrid agents."""
        return self._hybrid_agents.copy()
    
    def get_agents_by_category(self, category: str) -> Dict[str, Type[MicroAgent]]:
        """Get agents by category (based on naming patterns)."""
        category_lower = category.lower()
        matching_agents = {}
        
        for name, agent_class in self._agents.items():
            if category_lower in name.lower():
                matching_agents[name] = agent_class
        
        return matching_agents
    
    def create_agent(self, name: str, config: Optional[Dict] = None) -> Optional[MicroAgent]:
        """Create an instance of an agent by name."""
        agent_class = self.get_agent(name)
        if agent_class:
            return agent_class(config=config)
        return None
    
    def get_agent_info(self, name: str) -> Optional[Dict]:
        """Get detailed information about an agent."""
        agent_class = self.get_agent(name)
        if not agent_class:
            return None
        
        return {
            "name": name,
            "class": agent_class.__name__,
            "module": agent_class.__module__,
            "description": getattr(agent_class, '__doc__', 'No description'),
            "is_hybrid": issubclass(agent_class, HybridAgent),
            "input_model": getattr(agent_class, 'input_model', None),
            "output_model": getattr(agent_class, 'output_model', None)
        }
    
    def reload_agents(self):
        """Reload all agents from the filesystem."""
        self._agents.clear()
        self._hybrid_agents.clear()
        self._discover_agents()

# Decorator for automatic registration
def register_agent(registry_instance: AgentRegistry = None):
    """Decorator to automatically register agents."""
    def decorator(agent_class):
        registry = registry_instance or AgentRegistry()
        registry.register(agent_class)
        return agent_class
    return decorator

# Global registry instance
agent_registry = AgentRegistry()
