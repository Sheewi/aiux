"""
Microagent-Tools Integration Configuration

This module provides the configuration and interfaces needed to integrate
the comprehensive tools system with the microagents ecosystem.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import logging

# Import our tools system
from .base_tool import BaseTool, ToolMetadata, ToolStatus
from .registry import ToolsRegistry, ToolsManager

logger = logging.getLogger(__name__)


class MicroagentToolInterface:
    """
    Interface for microagents to interact with the tools system.
    Provides a simplified API for tool discovery and execution.
    """
    
    def __init__(self, registry: ToolsRegistry, manager: ToolsManager):
        self.registry = registry
        self.manager = manager
        self.agent_id = None
        self.context = {}
    
    def set_agent_context(self, agent_id: str, context: Dict[str, Any] = None):
        """Set the current agent context."""
        self.agent_id = agent_id
        self.context = context or {}
        logger.info(f"Set agent context for {agent_id}")
    
    def discover_tools(self, 
                      category: Optional[str] = None,
                      capability: Optional[str] = None,
                      query: Optional[str] = None) -> List[str]:
        """
        Discover available tools based on criteria.
        
        Args:
            category: Tool category to filter by
            capability: Required capability
            query: Search query for tool discovery
            
        Returns:
            List of tool names matching criteria
        """
        if query:
            return self.registry.search_tools(query)
        
        tools = []
        for tool_name in self.registry.list_tools():
            tool = self.registry.get_tool(tool_name)
            if not tool:
                continue
                
            metadata = tool.get_metadata()
            
            # Filter by category
            if category and metadata.tool_type.value != category:
                continue
                
            # Filter by capability
            if capability and capability not in [cap.value for cap in metadata.capabilities]:
                continue
                
            tools.append(tool_name)
        
        return tools
    
    async def execute_tool(self, 
                          tool_name: str,
                          operation: str,
                          parameters: Dict[str, Any],
                          timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a tool operation with microagent context.
        
        Args:
            tool_name: Name of the tool to execute
            operation: Operation to perform
            parameters: Operation parameters
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        # Add agent context to parameters if available
        if self.agent_id:
            parameters = parameters.copy()
            parameters['_agent_id'] = self.agent_id
            parameters['_agent_context'] = self.context
        
        # Execute the tool
        result = await self.manager.execute_tool(
            tool_name, operation, timeout=timeout, **parameters
        )
        
        # Log execution for the agent
        logger.info(f"Agent {self.agent_id} executed {tool_name}.{operation}: "
                   f"{'Success' if result['success'] else 'Failed'}")
        
        return result
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a tool."""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return None
        
        metadata = tool.get_metadata()
        return {
            'name': metadata.name,
            'description': metadata.description,
            'version': metadata.version,
            'category': metadata.tool_type.value,
            'capabilities': [cap.value for cap in metadata.capabilities],
            'parameters': metadata.input_schema,
            'output_schema': metadata.output_schema,
            'examples': getattr(metadata, 'examples', [])
        }
    
    def get_tool_categories(self) -> Dict[str, List[str]]:
        """Get all tool categories and their tools."""
        categories = {}
        for tool_name in self.registry.list_tools():
            tool = self.registry.get_tool(tool_name)
            if tool:
                category = tool.get_metadata().tool_type.value
                if category not in categories:
                    categories[category] = []
                categories[category].append(tool_name)
        return categories


class ToolsIntegrationManager:
    """
    Manages the integration between tools and microagents.
    Handles registration, discovery, and execution coordination.
    """
    
    def __init__(self, 
                 registry: ToolsRegistry, 
                 manager: ToolsManager,
                 microagent_registry_path: Optional[str] = None):
        self.registry = registry
        self.manager = manager
        self.microagent_registry_path = microagent_registry_path
        self.agent_interfaces = {}  # agent_id -> MicroagentToolInterface
        self.integration_config = {}
        
    def create_agent_interface(self, agent_id: str) -> MicroagentToolInterface:
        """Create a tool interface for a specific microagent."""
        interface = MicroagentToolInterface(self.registry, self.manager)
        interface.set_agent_context(agent_id)
        self.agent_interfaces[agent_id] = interface
        return interface
    
    def get_agent_interface(self, agent_id: str) -> Optional[MicroagentToolInterface]:
        """Get the tool interface for a specific microagent."""
        return self.agent_interfaces.get(agent_id)
    
    def register_tool_for_agent(self, 
                               agent_id: str, 
                               tool_name: str,
                               custom_config: Optional[Dict[str, Any]] = None):
        """Register a tool for use by a specific microagent."""
        if agent_id not in self.integration_config:
            self.integration_config[agent_id] = {
                'allowed_tools': [],
                'tool_configs': {},
                'permissions': {}
            }
        
        config = self.integration_config[agent_id]
        if tool_name not in config['allowed_tools']:
            config['allowed_tools'].append(tool_name)
        
        if custom_config:
            config['tool_configs'][tool_name] = custom_config
        
        logger.info(f"Registered tool {tool_name} for agent {agent_id}")
    
    def set_agent_permissions(self, 
                             agent_id: str,
                             permissions: Dict[str, Any]):
        """Set permissions for an agent's tool usage."""
        if agent_id not in self.integration_config:
            self.integration_config[agent_id] = {
                'allowed_tools': [],
                'tool_configs': {},
                'permissions': {}
            }
        
        self.integration_config[agent_id]['permissions'] = permissions
        logger.info(f"Set permissions for agent {agent_id}")
    
    def can_agent_use_tool(self, agent_id: str, tool_name: str) -> bool:
        """Check if an agent is allowed to use a specific tool."""
        config = self.integration_config.get(agent_id, {})
        allowed_tools = config.get('allowed_tools', [])
        
        # If no specific tools are configured, allow all
        if not allowed_tools:
            return True
        
        return tool_name in allowed_tools
    
    async def execute_tool_for_agent(self,
                                   agent_id: str,
                                   tool_name: str,
                                   operation: str,
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on behalf of a microagent with permission checks."""
        # Check permissions
        if not self.can_agent_use_tool(agent_id, tool_name):
            return {
                'success': False,
                'error': f'Agent {agent_id} is not authorized to use tool {tool_name}',
                'status': ToolStatus.FAILED
            }
        
        # Get or create agent interface
        interface = self.agent_interfaces.get(agent_id)
        if not interface:
            interface = self.create_agent_interface(agent_id)
        
        # Execute the tool
        return await interface.execute_tool(tool_name, operation, parameters)
    
    def index_tools_in_microagent_registry(self) -> bool:
        """Index all tools in the microagent registry."""
        if not self.microagent_registry_path:
            logger.warning("No microagent registry path configured")
            return False
        
        try:
            registry_file = Path(self.microagent_registry_path)
            
            # Load existing registry
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    microagent_registry = json.load(f)
            else:
                microagent_registry = {
                    'microagents': {},
                    'tools': {},
                    'integrations': {}
                }
            
            # Add tools to the registry
            tools_info = {}
            for tool_name in self.registry.list_tools():
                tool = self.registry.get_tool(tool_name)
                if tool:
                    metadata = tool.get_metadata()
                    tools_info[tool_name] = {
                        'name': metadata.name,
                        'description': metadata.description,
                        'version': metadata.version,
                        'category': metadata.tool_type.value,
                        'capabilities': [cap.value for cap in metadata.capabilities],
                        'parameters': metadata.input_schema,
                        'output_schema': metadata.output_schema,
                        'status': 'available',
                        'integration_type': 'tools_system'
                    }
            
            microagent_registry['tools'] = tools_info
            
            # Add integration information
            microagent_registry['integrations']['tools_system'] = {
                'manager_class': 'ToolsIntegrationManager',
                'registry_class': 'ToolsRegistry',
                'interface_class': 'MicroagentToolInterface',
                'total_tools': len(tools_info),
                'categories': list(set(t['category'] for t in tools_info.values())),
                'status': 'active'
            }
            
            # Save updated registry
            with open(registry_file, 'w') as f:
                json.dump(microagent_registry, f, indent=2, default=str)
            
            logger.info(f"Indexed {len(tools_info)} tools in microagent registry")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index tools in microagent registry: {e}")
            return False
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate a comprehensive integration report."""
        return {
            'timestamp': str(Path().resolve()),
            'tools_system': {
                'total_tools': len(self.registry.list_tools()),
                'categories': len(set(
                    tool.get_metadata().tool_type.value 
                    for tool in self.registry.tools.values()
                )),
                'status': 'operational'
            },
            'microagent_integration': {
                'registered_agents': len(self.agent_interfaces),
                'agent_configurations': len(self.integration_config),
                'registry_indexed': bool(self.microagent_registry_path)
            },
            'performance': self.manager.get_tool_performance_stats(),
            'configuration': {
                'agent_configs': self.integration_config,
                'registry_path': self.microagent_registry_path
            }
        }


# Integration configuration templates
INTEGRATION_TEMPLATES = {
    'basic_agent': {
        'allowed_tools': ['file_operations', 'data_processing', 'monitoring'],
        'permissions': {
            'max_concurrent_executions': 3,
            'timeout_seconds': 30,
            'rate_limit_rpm': 60
        }
    },
    'security_agent': {
        'allowed_tools': ['security', 'monitoring', 'file_operations'],
        'permissions': {
            'max_concurrent_executions': 2,
            'timeout_seconds': 60,
            'rate_limit_rpm': 30
        }
    },
    'web_agent': {
        'allowed_tools': ['tavily_search', 'web_scraping', 'api_clients', 'data_processing'],
        'permissions': {
            'max_concurrent_executions': 5,
            'timeout_seconds': 45,
            'rate_limit_rpm': 100
        }
    },
    'automation_agent': {
        'allowed_tools': ['automation', 'file_operations', 'monitoring', 'data_processing'],
        'permissions': {
            'max_concurrent_executions': 3,
            'timeout_seconds': 120,
            'rate_limit_rpm': 30
        }
    }
}


def create_integration_config(agent_type: str = 'basic_agent') -> Dict[str, Any]:
    """Create an integration configuration for a specific agent type."""
    return INTEGRATION_TEMPLATES.get(agent_type, INTEGRATION_TEMPLATES['basic_agent']).copy()


def setup_microagent_integration(registry: ToolsRegistry,
                                manager: ToolsManager,
                                microagent_registry_path: str = None) -> ToolsIntegrationManager:
    """
    Set up the complete integration between tools and microagents.
    
    Args:
        registry: The tools registry instance
        manager: The tools manager instance
        microagent_registry_path: Path to the microagent registry file
        
    Returns:
        Configured integration manager
    """
    # Default path if not provided
    if not microagent_registry_path:
        microagent_registry_path = str(Path(__file__).parent.parent / 'microagent_registry.json')
    
    # Create integration manager
    integration_manager = ToolsIntegrationManager(
        registry, manager, microagent_registry_path
    )
    
    # Index tools in microagent registry
    integration_manager.index_tools_in_microagent_registry()
    
    logger.info("Microagent-tools integration setup completed")
    return integration_manager
