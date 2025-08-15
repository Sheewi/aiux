"""
Comprehensive Tools System for Microagents Conversational AI

This package provides a complete toolkit of specialized tools for various operations
including web search, API integration, file operations, data processing, security,
automation, and monitoring.

Tool Categories:
- Search & Web: Tavily search, web scraping, sitemap extraction
- API Clients: REST, GraphQL, Webhooks, OpenAI integration
- File Operations: File management, configuration handling
- Data Processing: Data transformation, analysis, cleaning
- Security: Encryption, authentication, vulnerability scanning
- Automation: Task scheduling, workflow management
- Monitoring: System metrics, performance monitoring, health checks

Usage:
    from tools import global_registry, global_manager
    
    # List available tools
    tools = global_registry.list_tools()
    
    # Execute a tool
    result = await global_manager.execute_tool('tavily_search', 'search', query='AI news')
    
    # Execute tool chain
    chain = [
        {'tool': 'tavily_search', 'operation': 'search', 'parameters': {'query': 'AI news'}},
        {'tool': 'data_processing', 'operation': 'analyze', 'parameters': {'data': '${step_0.result.articles}'}}
    ]
    result = await global_manager.execute_tool_chain(chain)
"""

from .base_tool import BaseTool, ToolStatus, ToolMetadata

# Import all tool classes
from .tavily_search import TavilySearchTool, TavilyNewsSearchTool
from .web_scraping import WebScrapingTool, SitemapExtractorTool
from .api_clients import RestApiTool, GraphQLTool, WebhookTool, OpenAITool
from .file_operations import FileOperationsTool, ConfigurationTool
from .data_processing import DataProcessingTool
from .security import SecurityTool
from .automation import AutomationTool
from .monitoring import MonitoringTool

# Import registry and manager
from .registry import ToolsRegistry, ToolsManager, global_registry, global_manager

# Import integration utilities
try:
    from .integration import (
        MicroagentToolInterface,
        ToolsIntegrationManager,
        setup_microagent_integration,
        create_integration_config,
        INTEGRATION_TEMPLATES
    )
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Integration utilities not available: {e}")
    MicroagentToolInterface = None
    ToolsIntegrationManager = None
    setup_microagent_integration = None
    create_integration_config = None
    INTEGRATION_TEMPLATES = None

# Tool categories for easy discovery
TOOL_CATEGORIES = {
    'search_web': {
        'description': 'Web search and content extraction tools',
        'tools': ['tavily_search', 'tavily_news_search', 'web_scraping', 'sitemap_extractor']
    },
    'api_clients': {
        'description': 'API integration and external service clients',
        'tools': ['rest_api', 'graphql_api', 'webhook', 'openai_api']
    },
    'file_operations': {
        'description': 'File system operations and configuration management',
        'tools': ['file_operations', 'configuration']
    },
    'data_processing': {
        'description': 'Data transformation, analysis, and processing',
        'tools': ['data_processing']
    },
    'security': {
        'description': 'Security, encryption, and vulnerability assessment',
        'tools': ['security']
    },
    'automation': {
        'description': 'Task scheduling and workflow automation',
        'tools': ['automation']
    },
    'monitoring': {
        'description': 'System monitoring and performance analysis',
        'tools': ['monitoring']
    }
}

# Quick access functions
def list_all_tools():
    """List all available tools."""
    return global_registry.list_tools()

def get_tool_categories():
    """Get all tool categories."""
    return TOOL_CATEGORIES

def search_tools(query: str):
    """Search for tools by name, description, or capabilities."""
    return global_registry.search_tools(query)

def get_tool_info(tool_name: str):
    """Get detailed information about a specific tool."""
    metadata = global_registry.get_tool_metadata(tool_name)
    if metadata:
        return {
            'name': metadata.name,
            'description': metadata.description,
            'version': metadata.version,
            'author': metadata.author,
            'tags': metadata.tags,
            'capabilities': metadata.capabilities,
            'parameters': metadata.parameters
        }
    return None

async def quick_execute(tool_name: str, operation: str, **kwargs):
    """Quick tool execution function."""
    return await global_manager.execute_tool(tool_name, operation, **kwargs)

def create_tool(tool_name: str, **kwargs):
    """Create a new instance of a tool."""
    return global_registry.create_tool(tool_name, **kwargs)

def validate_tool_config(tool_name: str, config: dict):
    """Validate configuration for a tool."""
    return global_registry.validate_tool_config(tool_name, config)

# Tool execution examples
TOOL_EXAMPLES = {
    'tavily_search': {
        'description': 'Search the web for information',
        'example': {
            'tool': 'tavily_search',
            'operation': 'search',
            'parameters': {
                'query': 'latest AI developments',
                'max_results': 5,
                'include_images': True
            }
        }
    },
    'web_scraping': {
        'description': 'Extract content from web pages',
        'example': {
            'tool': 'web_scraping',
            'operation': 'scrape',
            'parameters': {
                'url': 'https://example.com',
                'extract_links': True,
                'extract_images': True
            }
        }
    },
    'file_operations': {
        'description': 'Perform file system operations',
        'example': {
            'tool': 'file_operations',
            'operation': 'read',
            'parameters': {
                'file_path': '/path/to/file.txt',
                'encoding': 'utf-8'
            }
        }
    },
    'data_processing': {
        'description': 'Process and analyze data',
        'example': {
            'tool': 'data_processing',
            'operation': 'analyze',
            'parameters': {
                'data': {'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']},
                'analysis_type': 'summary'
            }
        }
    },
    'security': {
        'description': 'Security operations and analysis',
        'example': {
            'tool': 'security',
            'operation': 'hash',
            'parameters': {
                'data': 'password123',
                'algorithm': 'sha256',
                'salt': 'random_salt'
            }
        }
    },
    'automation': {
        'description': 'Automate tasks and workflows',
        'example': {
            'tool': 'automation',
            'operation': 'schedule_task',
            'parameters': {
                'task_id': 'daily_backup',
                'task_type': 'command',
                'schedule_type': 'daily',
                'schedule_config': {'time': '02:00'},
                'task_config': {'command': 'backup.sh'}
            }
        }
    },
    'monitoring': {
        'description': 'Monitor system metrics and performance',
        'example': {
            'tool': 'monitoring',
            'operation': 'system_metrics',
            'parameters': {}
        }
    }
}

def get_tool_examples():
    """Get example usage for all tools."""
    return TOOL_EXAMPLES

def get_tool_example(tool_name: str):
    """Get example usage for a specific tool."""
    return TOOL_EXAMPLES.get(tool_name)

# Export all important classes and functions
__all__ = [
    # Base classes
    'BaseTool', 'ToolStatus', 'ToolMetadata',
    
    # Tool classes
    'TavilySearchTool', 'TavilyNewsSearchTool',
    'WebScrapingTool', 'SitemapExtractorTool',
    'RestApiTool', 'GraphQLTool', 'WebhookTool', 'OpenAITool',
    'FileOperationsTool', 'ConfigurationTool',
    'DataProcessingTool',
    'SecurityTool',
    'AutomationTool',
    'MonitoringTool',
    
    # Registry and manager
    'ToolsRegistry', 'ToolsManager',
    'global_registry', 'global_manager',
    
    # Integration components
    'MicroagentToolInterface', 'ToolsIntegrationManager',
    'setup_microagent_integration', 'create_integration_config',
    'INTEGRATION_TEMPLATES',
    
    # Utility functions
    'list_all_tools', 'get_tool_categories', 'search_tools',
    'get_tool_info', 'quick_execute', 'create_tool',
    'validate_tool_config', 'get_tool_examples', 'get_tool_example',
    
    # Constants
    'TOOL_CATEGORIES', 'TOOL_EXAMPLES'
]

# Version information
__version__ = '1.0.0'
__author__ = 'AI Assistant'
__description__ = 'Comprehensive tools system for microagents conversational AI'
