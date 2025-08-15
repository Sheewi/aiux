"""
Agent Registry System
Dynamic agent discovery and management system for the microagent ecosystem.
"""

import os
import importlib
import inspect
from typing import Dict, List, Type, Optional, Any
from pathlib import Path

# Try to import base agent classes
try:
    from generated_agents.base_agent import MicroAgent, HybridAgent
except ImportError:
    try:
        from generated_agents.core.base_agent import MicroAgent, HybridAgent
    except ImportError:
        # Fallback: create minimal base classes if not found
        class MicroAgent:
            """Fallback base agent class."""
            def __init__(self, config=None):
                self.config = config or {}
        
        class HybridAgent(MicroAgent):
            """Fallback hybrid agent class."""
            pass

class AgentRegistry:
    """
    Central registry for discovering and managing agents.
    Singleton pattern ensures consistent agent discovery across the system.
    """
    
    _instance = None
    _agents: Dict[str, Type[MicroAgent]]
    _hybrid_agents: Dict[str, Type[HybridAgent]]
    _agent_metadata: Dict[str, dict]
    _initialized: bool
    
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
            metadata: Dict with keys: 'capabilities', 'token_formats', 'resource_footprint', 'input_modes', 'output_modes'
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
                'resource_footprint': getattr(agent_class, 'resource_footprint', {}),
                'input_modes': getattr(agent_class, 'input_modes', ['natural_language']),
                'output_modes': getattr(agent_class, 'output_modes', ['text']),
                'intent_keywords': getattr(agent_class, 'intent_keywords', []),
                'multimodal_capable': getattr(agent_class, 'multimodal_capable', False)
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
    
    def get_agents_by_intent(self, intent: str, input_mode: str = "natural_language", 
                           output_mode: str = "text") -> Dict[str, Type[MicroAgent]]:
        """Get agents that can handle a specific intent with given input/output modes."""
        matching_agents = {}
        intent_lower = intent.lower()
        
        for name, agent_class in self._agents.items():
            metadata = self._agent_metadata.get(name, {})
            
            # Check intent keywords
            intent_keywords = metadata.get('intent_keywords', [])
            if any(keyword.lower() in intent_lower for keyword in intent_keywords):
                # Check input/output mode compatibility
                input_modes = metadata.get('input_modes', ['natural_language'])
                output_modes = metadata.get('output_modes', ['text'])
                
                if input_mode in input_modes and output_mode in output_modes:
                    matching_agents[name] = agent_class
            
            # Also check name-based matching as fallback
            elif intent_lower in name.lower():
                matching_agents[name] = agent_class
        
        return matching_agents
    
    def get_multimodal_capable_agents(self) -> Dict[str, Type[MicroAgent]]:
        """Get all agents that support multimodal input/output."""
        multimodal_agents = {}
        
        for name, agent_class in self._agents.items():
            metadata = self._agent_metadata.get(name, {})
            if metadata.get('multimodal_capable', False):
                multimodal_agents[name] = agent_class
        
        return multimodal_agents
    
    def suggest_agents_for_context(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest agents based on multimodal context."""
        suggestions = []
        
        intent = context.get('intent', 'general')
        entities = context.get('entities', [])
        input_mode = context.get('input_mode', 'natural_language')
        suggested_output_mode = context.get('suggested_output_mode', 'text')
        
        # Get agents by intent
        intent_agents = self.get_agents_by_intent(intent, input_mode, suggested_output_mode)
        
        for name, agent_class in intent_agents.items():
            metadata = self._agent_metadata.get(name, {})
            
            # Calculate relevance score
            relevance_score = self._calculate_agent_relevance(metadata, context)
            
            suggestions.append({
                'agent_name': name,
                'agent_class': agent_class,
                'relevance_score': relevance_score,
                'metadata': metadata,
                'reason': self._explain_suggestion(metadata, context)
            })
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return suggestions
    
    def _calculate_agent_relevance(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how relevant an agent is for the given context."""
        score = 0.0
        
        # Intent keyword matching
        intent = context.get('intent', '').lower()
        intent_keywords = metadata.get('intent_keywords', [])
        keyword_matches = sum(1 for keyword in intent_keywords if keyword.lower() in intent)
        score += keyword_matches * 0.3
        
        # Entity type matching
        entities = context.get('entities', [])
        entity_types = [e.get('type', '') for e in entities]
        capabilities = metadata.get('capabilities', [])
        capability_matches = sum(1 for cap in capabilities if any(et in cap.lower() for et in entity_types))
        score += capability_matches * 0.2
        
        # Mode compatibility
        input_mode = context.get('input_mode', 'natural_language')
        suggested_output_mode = context.get('suggested_output_mode', 'text')
        
        input_modes = metadata.get('input_modes', ['natural_language'])
        output_modes = metadata.get('output_modes', ['text'])
        
        if input_mode in input_modes:
            score += 0.2
        if suggested_output_mode in output_modes:
            score += 0.2
        
        # Multimodal bonus
        if metadata.get('multimodal_capable', False) and len(input_modes) > 1:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _explain_suggestion(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate explanation for why an agent was suggested."""
        reasons = []
        
        # Intent matching
        intent = context.get('intent', '').lower()
        intent_keywords = metadata.get('intent_keywords', [])
        matching_keywords = [k for k in intent_keywords if k.lower() in intent]
        if matching_keywords:
            reasons.append(f"matches intent keywords: {', '.join(matching_keywords)}")
        
        # Capability matching
        entities = context.get('entities', [])
        entity_types = [e.get('type', '') for e in entities]
        capabilities = metadata.get('capabilities', [])
        matching_caps = [cap for cap in capabilities if any(et in cap.lower() for et in entity_types)]
        if matching_caps:
            reasons.append(f"has relevant capabilities: {', '.join(matching_caps[:2])}")
        
        # Mode compatibility
        input_mode = context.get('input_mode', 'natural_language')
        suggested_output_mode = context.get('suggested_output_mode', 'text')
        if input_mode in metadata.get('input_modes', []):
            reasons.append(f"supports {input_mode} input")
        if suggested_output_mode in metadata.get('output_modes', []):
            reasons.append(f"supports {suggested_output_mode} output")
        
        if not reasons:
            return "general purpose compatibility"
        
        return "; ".join(reasons)

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
