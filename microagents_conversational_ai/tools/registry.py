import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime
from pathlib import Path
import importlib
import inspect

from .base_tool import BaseTool, ToolMetadata, ToolStatus
from .tavily_search import TavilySearchTool, TavilyNewsSearchTool
from .web_scraping import WebScrapingTool, SitemapExtractorTool
from .api_clients import RestApiTool, GraphQLTool, WebhookTool, OpenAITool
from .file_operations import FileOperationsTool, ConfigurationTool
from .data_processing import DataProcessingTool
from .security import SecurityTool
from .automation import AutomationTool
from .monitoring import MonitoringTool


class ToolsRegistry:
    """Centralized registry for managing all available tools."""
    
    def __init__(self):
        """Initialize the tools registry."""
        self.logger = logging.getLogger(__name__)
        self.tools: Dict[str, BaseTool] = {}
        self.tool_classes: Dict[str, Type[BaseTool]] = {}
        self.tool_metadata: Dict[str, ToolMetadata] = {}
        self.tool_categories: Dict[str, List[str]] = {}
        self.load_default_tools()
    
    def load_default_tools(self):
        """Load all default tools into the registry."""
        # Define tool mappings
        default_tools = {
            # Search and Web Tools
            'tavily_search': TavilySearchTool,
            'tavily_news_search': TavilyNewsSearchTool,
            'web_scraping': WebScrapingTool,
            'sitemap_extractor': SitemapExtractorTool,
            
            # API Client Tools
            'rest_api': RestApiTool,
            'graphql_api': GraphQLTool,
            'webhook': WebhookTool,
            'openai_api': OpenAITool,
            
            # File Operations Tools
            'file_operations': FileOperationsTool,
            'configuration': ConfigurationTool,
            
            # Data Processing Tools
            'data_processing': DataProcessingTool,
            
            # Security Tools
            'security': SecurityTool,
            
            # Automation Tools
            'automation': AutomationTool,
            
            # Monitoring Tools
            'monitoring': MonitoringTool
        }
        
        # Register all tools
        for tool_name, tool_class in default_tools.items():
            self.register_tool_class(tool_name, tool_class)
        
        # Define tool categories
        self.tool_categories = {
            'search_web': ['tavily_search', 'tavily_news_search', 'web_scraping', 'sitemap_extractor'],
            'api_clients': ['rest_api', 'graphql_api', 'webhook', 'openai_api'],
            'file_operations': ['file_operations', 'configuration'],
            'data_processing': ['data_processing'],
            'security': ['security'],
            'automation': ['automation'],
            'monitoring': ['monitoring']
        }
        
        self.logger.info(f"Loaded {len(default_tools)} default tools into registry")
    
    def register_tool_class(self, name: str, tool_class: Type[BaseTool]):
        """Register a tool class in the registry."""
        if not issubclass(tool_class, BaseTool):
            raise ValueError(f"Tool class {tool_class.__name__} must inherit from BaseTool")
        
        self.tool_classes[name] = tool_class
        
        # Get metadata from tool class if available
        try:
            instance = tool_class()
            metadata = instance.get_metadata()
            self.tool_metadata[name] = metadata
        except Exception as e:
            self.logger.warning(f"Could not get metadata for tool {name}: {e}")
            # Create basic metadata with proper parameters
            from .base_tool import ToolType
            self.tool_metadata[name] = ToolMetadata(
                tool_id=name,
                name=name,
                description=f"Tool class {tool_class.__name__}",
                tool_type=ToolType.CUSTOM,
                version="1.0.0",
                author="Unknown",
                capabilities=[],
                input_schema={},
                output_schema={}
            )
        self.logger.debug(f"Registered tool class: {name}")
    
    def register_tool_instance(self, name: str, tool_instance: BaseTool):
        """Register a tool instance in the registry."""
        if not isinstance(tool_instance, BaseTool):
            raise ValueError(f"Tool instance must inherit from BaseTool")
        
        self.tools[name] = tool_instance
        self.tool_metadata[name] = tool_instance.get_metadata()
        
        self.logger.debug(f"Registered tool instance: {name}")
    
    def create_tool(self, name: str, **kwargs) -> BaseTool:
        """Create a new instance of a tool."""
        if name not in self.tool_classes:
            raise ValueError(f"Unknown tool: {name}")
        
        tool_class = self.tool_classes[name]
        return tool_class(**kwargs)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        if name in self.tools:
            return self.tools[name]
        elif name in self.tool_classes:
            # Create instance on demand
            tool_instance = self.create_tool(name)
            self.tools[name] = tool_instance
            return tool_instance
        return None
    
    def list_tools(self, category: Optional[str] = None, 
                  capability: Optional[str] = None) -> List[str]:
        """List available tools, optionally filtered by category or capability."""
        if category:
            return self.tool_categories.get(category, [])
        
        if capability:
            matching_tools = []
            for tool_name, metadata in self.tool_metadata.items():
                if capability in metadata.capabilities:
                    matching_tools.append(tool_name)
            return matching_tools
        
        return list(self.tool_classes.keys())
    
    def get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool."""
        return self.tool_metadata.get(name)
    
    def get_all_metadata(self) -> Dict[str, ToolMetadata]:
        """Get metadata for all registered tools."""
        return self.tool_metadata.copy()
    
    def get_categories(self) -> Dict[str, List[str]]:
        """Get all tool categories."""
        return self.tool_categories.copy()
    
    def search_tools(self, query: str) -> List[str]:
        """Search for tools by name, description, or capabilities."""
        query_lower = query.lower()
        matching_tools = []
        
        for tool_name, metadata in self.tool_metadata.items():
            # Search in name
            if query_lower in tool_name.lower():
                matching_tools.append(tool_name)
                continue
            
            # Search in description
            if query_lower in metadata.description.lower():
                matching_tools.append(tool_name)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in metadata.tags):
                matching_tools.append(tool_name)
                continue
            
            # Search in capabilities
            if any(query_lower in cap.lower() for cap in metadata.capabilities):
                matching_tools.append(tool_name)
                continue
        
        return matching_tools
    
    def get_tool_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all tools."""
        stats = {}
        
        for tool_name, tool_instance in self.tools.items():
            stats[tool_name] = {
                'status': tool_instance.status.value if hasattr(tool_instance.status, 'value') else str(tool_instance.status),
                'created_at': getattr(tool_instance, 'created_at', None),
                'last_used': getattr(tool_instance, 'last_used', None),
                'execution_count': getattr(tool_instance, 'execution_count', 0)
            }
        
        return stats
    
    def validate_tool_config(self, tool_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration for a specific tool."""
        metadata = self.get_tool_metadata(tool_name)
        if not metadata:
            return {'valid': False, 'error': f'Unknown tool: {tool_name}'}
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required parameters
        for param_name, param_info in metadata.parameters.items():
            if param_info.get('required', False) and param_name not in config:
                validation_result['errors'].append(f"Missing required parameter: {param_name}")
                validation_result['valid'] = False
        
        # Check parameter types and constraints
        for param_name, value in config.items():
            if param_name in metadata.parameters:
                param_info = metadata.parameters[param_name]
                
                # Type checking (basic)
                expected_type = param_info.get('type')
                if expected_type == 'string' and not isinstance(value, str):
                    validation_result['errors'].append(f"Parameter {param_name} should be string")
                    validation_result['valid'] = False
                elif expected_type == 'integer' and not isinstance(value, int):
                    validation_result['errors'].append(f"Parameter {param_name} should be integer")
                    validation_result['valid'] = False
                elif expected_type == 'boolean' and not isinstance(value, bool):
                    validation_result['errors'].append(f"Parameter {param_name} should be boolean")
                    validation_result['valid'] = False
                
                # Enum checking
                if 'enum' in param_info and value not in param_info['enum']:
                    validation_result['errors'].append(f"Parameter {param_name} must be one of: {param_info['enum']}")
                    validation_result['valid'] = False
            else:
                validation_result['warnings'].append(f"Unknown parameter: {param_name}")
        
        return validation_result
    
    def export_registry(self, file_path: str):
        """Export the registry to a JSON file."""
        registry_data = {
            'tools': list(self.tool_classes.keys()),
            'categories': self.tool_categories,
            'metadata': {
                name: {
                    'name': metadata.name,
                    'description': metadata.description,
                    'version': metadata.version,
                    'author': metadata.author,
                    'tags': metadata.tags,
                    'capabilities': metadata.capabilities,
                    'parameters': metadata.parameters
                }
                for name, metadata in self.tool_metadata.items()
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        self.logger.info(f"Registry exported to {file_path}")
    
    def import_registry(self, file_path: str):
        """Import registry data from a JSON file."""
        with open(file_path, 'r') as f:
            registry_data = json.load(f)
        
        # Update categories
        self.tool_categories.update(registry_data.get('categories', {}))
        
        # Update metadata (for external tools)
        for tool_name, metadata_dict in registry_data.get('metadata', {}).items():
            if tool_name not in self.tool_metadata:
                self.tool_metadata[tool_name] = ToolMetadata(**metadata_dict)
        
        self.logger.info(f"Registry imported from {file_path}")


class ToolsManager:
    """High-level manager for tool execution and coordination."""
    
    def __init__(self, registry: Optional[ToolsRegistry] = None):
        """Initialize the tools manager."""
        self.registry = registry or ToolsRegistry()
        self.logger = logging.getLogger(__name__)
        self.active_tools: Dict[str, BaseTool] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute_tool(self, tool_name: str, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool operation."""
        execution_id = f"{tool_name}_{operation}_{int(datetime.now().timestamp())}"
        
        try:
            # Get or create tool instance
            tool = self.registry.get_tool(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            
            # Track active tool
            self.active_tools[execution_id] = tool
            
            # Execute the tool
            start_time = datetime.now()
            result = await tool.execute(operation, **kwargs)
            end_time = datetime.now()
            
            # Record execution
            execution_record = {
                'execution_id': execution_id,
                'tool_name': tool_name,
                'operation': operation,
                'parameters': kwargs,
                'result': result,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'success': result.get('success', False)
            }
            
            self.execution_history.append(execution_record)
            
            # Remove from active tools
            self.active_tools.pop(execution_id, None)
            
            return execution_record
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            
            # Remove from active tools
            self.active_tools.pop(execution_id, None)
            
            execution_record = {
                'execution_id': execution_id,
                'tool_name': tool_name,
                'operation': operation,
                'parameters': kwargs,
                'error': str(e),
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 0,
                'success': False
            }
            
            self.execution_history.append(execution_record)
            return execution_record
    
    async def execute_tool_chain(self, chain_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a chain of tool operations."""
        chain_id = f"chain_{int(datetime.now().timestamp())}"
        chain_results = []
        chain_context = {}
        
        start_time = datetime.now()
        
        try:
            for step_index, step_config in enumerate(chain_config):
                tool_name = step_config.get('tool')
                operation = step_config.get('operation')
                parameters = step_config.get('parameters', {})
                
                # Process parameter substitutions from context
                processed_params = self._process_parameters(parameters, chain_context)
                
                # Execute step
                step_result = await self.execute_tool(tool_name, operation, **processed_params)
                
                # Update context with step result
                context_key = step_config.get('context_key', f'step_{step_index}')
                chain_context[context_key] = step_result
                
                chain_results.append({
                    'step_index': step_index,
                    'tool_name': tool_name,
                    'operation': operation,
                    'result': step_result,
                    'success': step_result.get('success', False)
                })
                
                # Stop chain if step failed and fail_fast is enabled
                if not step_result.get('success', False) and step_config.get('fail_fast', False):
                    break
            
            end_time = datetime.now()
            
            # Calculate success rate
            successful_steps = sum(1 for result in chain_results if result['success'])
            total_steps = len(chain_results)
            
            return {
                'chain_id': chain_id,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'total_steps': total_steps,
                'successful_steps': successful_steps,
                'success_rate': successful_steps / total_steps if total_steps > 0 else 0,
                'results': chain_results,
                'context': chain_context,
                'overall_success': successful_steps == total_steps
            }
            
        except Exception as e:
            self.logger.error(f"Tool chain execution failed: {e}")
            return {
                'chain_id': chain_id,
                'error': str(e),
                'results': chain_results,
                'overall_success': False
            }
    
    def _process_parameters(self, parameters: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Process parameter substitutions from context."""
        processed = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # Parameter substitution
                context_path = value[2:-1]  # Remove ${ and }
                
                # Navigate context path (e.g., step_0.result.data)
                try:
                    result_value = context
                    for path_part in context_path.split('.'):
                        result_value = result_value[path_part]
                    processed[key] = result_value
                except (KeyError, TypeError):
                    self.logger.warning(f"Context path {context_path} not found, using original value")
                    processed[key] = value
            else:
                processed[key] = value
        
        return processed
    
    async def execute_parallel_tools(self, tool_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple tools in parallel."""
        parallel_id = f"parallel_{int(datetime.now().timestamp())}"
        
        start_time = datetime.now()
        
        # Create execution tasks
        tasks = []
        for config in tool_configs:
            tool_name = config.get('tool')
            operation = config.get('operation')
            parameters = config.get('parameters', {})
            
            task = self.execute_tool(tool_name, operation, **parameters)
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        
        # Process results
        processed_results = []
        successful_executions = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'tool_config': tool_configs[i],
                    'success': False,
                    'error': str(result)
                })
            else:
                processed_results.append({
                    'tool_config': tool_configs[i],
                    'result': result,
                    'success': result.get('success', False)
                })
                if result.get('success', False):
                    successful_executions += 1
        
        return {
            'parallel_id': parallel_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'total_executions': len(tool_configs),
            'successful_executions': successful_executions,
            'success_rate': successful_executions / len(tool_configs) if tool_configs else 0,
            'results': processed_results,
            'overall_success': successful_executions == len(tool_configs)
        }
    
    def get_execution_history(self, tool_name: Optional[str] = None,
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution history, optionally filtered by tool name."""
        history = self.execution_history
        
        if tool_name:
            history = [record for record in history if record['tool_name'] == tool_name]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_active_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently active tools."""
        active_info = {}
        
        for execution_id, tool in self.active_tools.items():
            active_info[execution_id] = {
                'tool_name': tool.__class__.__name__,
                'status': tool.status.value if hasattr(tool.status, 'value') else str(tool.status),
                'created_at': getattr(tool, 'created_at', None)
            }
        
        return active_info
    
    def get_tool_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for each tool."""
        stats = {}
        
        for record in self.execution_history:
            tool_name = record['tool_name']
            
            if tool_name not in stats:
                stats[tool_name] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'total_duration': 0,
                    'average_duration': 0,
                    'min_duration': float('inf'),
                    'max_duration': 0
                }
            
            tool_stats = stats[tool_name]
            tool_stats['total_executions'] += 1
            
            if record['success']:
                tool_stats['successful_executions'] += 1
            else:
                tool_stats['failed_executions'] += 1
            
            duration = record.get('duration_seconds', 0)
            tool_stats['total_duration'] += duration
            tool_stats['min_duration'] = min(tool_stats['min_duration'], duration)
            tool_stats['max_duration'] = max(tool_stats['max_duration'], duration)
            
            # Calculate average
            tool_stats['average_duration'] = tool_stats['total_duration'] / tool_stats['total_executions']
            
            # Calculate success rate
            tool_stats['success_rate'] = tool_stats['successful_executions'] / tool_stats['total_executions']
        
        return stats
    
    def clear_history(self, before_date: Optional[datetime] = None):
        """Clear execution history."""
        if before_date:
            self.execution_history = [
                record for record in self.execution_history
                if datetime.fromisoformat(record['start_time']) >= before_date
            ]
        else:
            self.execution_history.clear()
        
        self.logger.info("Execution history cleared")


# Global registry instance
global_registry = ToolsRegistry()
global_manager = ToolsManager(global_registry)
