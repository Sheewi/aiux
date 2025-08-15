# Microagent-Tools Integration Guide

## Overview

This guide describes the integration between the comprehensive tools system and microagents.

## Integration Status

- **Total Tools**: 14
- **Tool Categories**: 10
- **Registry Path**: /media/r/Workspace/microagents_conversational_ai/microagent_registry.json
- **Configured Agents**: 4

## Available Tools

## Configured Agents

### web_researcher
- **Type**: web_agent
- **Available Tools**: 14
- **Tools**: tavily_search, tavily_news_search, web_scraping, sitemap_extractor, rest_api, graphql_api, webhook, openai_api, file_operations, configuration, data_processing, security, automation, monitoring
- **Max Concurrent**: 5
- **Timeout**: 45s
- **Rate Limit**: 100 req/min

### security_auditor
- **Type**: security_agent
- **Available Tools**: 14
- **Tools**: tavily_search, tavily_news_search, web_scraping, sitemap_extractor, rest_api, graphql_api, webhook, openai_api, file_operations, configuration, data_processing, security, automation, monitoring
- **Max Concurrent**: 2
- **Timeout**: 60s
- **Rate Limit**: 30 req/min

### file_manager
- **Type**: basic_agent
- **Available Tools**: 14
- **Tools**: tavily_search, tavily_news_search, web_scraping, sitemap_extractor, rest_api, graphql_api, webhook, openai_api, file_operations, configuration, data_processing, security, automation, monitoring
- **Max Concurrent**: 3
- **Timeout**: 30s
- **Rate Limit**: 60 req/min

### automation_controller
- **Type**: automation_agent
- **Available Tools**: 14
- **Tools**: tavily_search, tavily_news_search, web_scraping, sitemap_extractor, rest_api, graphql_api, webhook, openai_api, file_operations, configuration, data_processing, security, automation, monitoring
- **Max Concurrent**: 3
- **Timeout**: 120s
- **Rate Limit**: 30 req/min

## Usage Examples

### Basic Tool Execution

```python
from tools import setup_microagent_integration, global_registry, global_manager

# Set up integration
integration_manager = setup_microagent_integration(
    global_registry, global_manager, 'microagent_registry.json'
)

# Create agent interface
interface = integration_manager.create_agent_interface('my_agent')

# Execute a tool
result = await interface.execute_tool(
    'security', 'hash',
    {'data': 'sensitive_data', 'algorithm': 'sha256'}
)
```

### Agent Configuration

```python
# Register tools for an agent
integration_manager.register_tool_for_agent('my_agent', 'security')
integration_manager.register_tool_for_agent('my_agent', 'file_operations')

# Set permissions
integration_manager.set_agent_permissions('my_agent', {
    'max_concurrent_executions': 5,
    'timeout_seconds': 30,
    'rate_limit_rpm': 100
})
```

### Tool Discovery

```python
# Get agent interface
interface = integration_manager.get_agent_interface('my_agent')

# Discover available tools
tools = interface.discover_tools()

# Search for specific tools
web_tools = interface.discover_tools(category='search_web')

# Get tool information
tool_info = interface.get_tool_info('security')
```

## Integration Architecture

The integration provides:

1. **MicroagentToolInterface**: Simplified API for tool interaction
2. **ToolsIntegrationManager**: Centralized integration management
3. **Permission System**: Fine-grained access control
4. **Registry Indexing**: Automatic tool discovery
5. **Performance Monitoring**: Execution tracking and statistics

## Next Steps

1. Configure your microagents using the integration manager
2. Set appropriate permissions for each agent
3. Use the agent interfaces for tool execution
4. Monitor performance using the built-in statistics

For more details, see the tools documentation and examples.
