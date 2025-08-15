#!/usr/bin/env python3
"""
Microagent-Tools Integration Setup Script

This script configures the integration between the tools system and microagents,
including registry indexing and example agent configurations.

Usage:
    python setup_integration.py [options]
    
Options:
    --registry-path PATH    Path to microagent registry file
    --config-agents        Configure example agents
    --demo                 Run integration demo
    --help                 Show this help message
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from tools import (
        global_registry, global_manager,
        setup_microagent_integration,
        create_integration_config,
        INTEGRATION_TEMPLATES
    )
except ImportError as e:
    print(f"Error importing tools system: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def find_microagent_registry() -> str:
    """Find the microagent registry file."""
    possible_paths = [
        Path(__file__).parent.parent / 'microagent_registry.json',
        Path(__file__).parent.parent / 'microagents_conversational_ai' / 'microagent_registry.json',
        Path(__file__).parent.parent / 'registry.py',  # Fallback to existing registry
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # Default path for new registry
    return str(Path(__file__).parent.parent / 'microagent_registry.json')


def setup_integration(registry_path: str = None) -> Dict[str, Any]:
    """Set up the complete microagent-tools integration."""
    if not registry_path:
        registry_path = find_microagent_registry()
    
    print(f"Setting up integration with registry: {registry_path}")
    
    # Create integration manager
    integration_manager = setup_microagent_integration(
        global_registry, global_manager, registry_path
    )
    
    # Generate integration report
    report = integration_manager.generate_integration_report()
    
    print(f"✓ Integration setup completed")
    print(f"  - Total tools: {report['tools_system']['total_tools']}")
    print(f"  - Tool categories: {report['tools_system']['categories']}")
    print(f"  - Registry path: {registry_path}")
    
    return {
        'integration_manager': integration_manager,
        'report': report,
        'registry_path': registry_path
    }


def configure_example_agents(integration_manager) -> Dict[str, Any]:
    """Configure example microagents with tool access."""
    print("\nConfiguring example agents...")
    
    agent_configs = {}
    
    # Configure different types of agents
    agent_types = {
        'web_researcher': 'web_agent',
        'security_auditor': 'security_agent', 
        'file_manager': 'basic_agent',
        'automation_controller': 'automation_agent'
    }
    
    for agent_id, agent_type in agent_types.items():
        # Get template configuration
        config = create_integration_config(agent_type)
        
        # Register the agent
        for tool_name in config['allowed_tools']:
            integration_manager.register_tool_for_agent(agent_id, tool_name)
        
        # Set permissions
        integration_manager.set_agent_permissions(agent_id, config['permissions'])
        
        # Create interface
        interface = integration_manager.create_agent_interface(agent_id)
        
        agent_configs[agent_id] = {
            'type': agent_type,
            'config': config,
            'interface': interface,
            'available_tools': interface.discover_tools()
        }
        
        print(f"  ✓ {agent_id} ({agent_type}): {len(config['allowed_tools'])} tools")
    
    return agent_configs


async def run_integration_demo(integration_manager, agent_configs):
    """Run a demonstration of the integration."""
    print("\n" + "="*60)
    print("INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Test each configured agent
    for agent_id, agent_info in agent_configs.items():
        print(f"\nTesting agent: {agent_id}")
        interface = agent_info['interface']
        
        # Discover tools
        tools = interface.discover_tools()
        print(f"  Available tools: {', '.join(tools[:3])}{'...' if len(tools) > 3 else ''}")
        
        # Test tool execution (using a safe tool)
        if 'security' in tools:
            print("  Testing security tool...")
            try:
                result = await interface.execute_tool(
                    'security', 'hash',
                    {'data': f'test_data_for_{agent_id}', 'algorithm': 'sha256'}
                )
                if result['success']:
                    print(f"    ✓ Hash generated successfully")
                else:
                    print(f"    ✗ Tool execution failed: {result.get('error')}")
            except Exception as e:
                print(f"    ✗ Tool execution error: {e}")
        
        elif 'monitoring' in tools:
            print("  Testing monitoring tool...")
            try:
                result = await interface.execute_tool(
                    'monitoring', 'system_metrics', {}
                )
                if result['success']:
                    metrics = result['result']['result']
                    cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
                    print(f"    ✓ System metrics retrieved (CPU: {cpu_usage:.1f}%)")
                else:
                    print(f"    ✗ Tool execution failed: {result.get('error')}")
            except Exception as e:
                print(f"    ✗ Tool execution error: {e}")
    
    # Test permission enforcement
    print(f"\nTesting permission enforcement...")
    web_agent = agent_configs.get('web_researcher')
    if web_agent:
        interface = web_agent['interface']
        
        # Try to use an unauthorized tool (should fail)
        try:
            result = await integration_manager.execute_tool_for_agent(
                'web_researcher', 'automation', 'schedule_task',
                {'task_id': 'test', 'task_type': 'command', 'schedule_type': 'once'}
            )
            if not result['success'] and 'not authorized' in result.get('error', ''):
                print("  ✓ Permission enforcement working correctly")
            else:
                print("  ⚠ Permission enforcement may not be working")
        except Exception as e:
            print(f"  ✗ Permission test error: {e}")
    
    # Show performance stats
    print(f"\nPerformance Statistics:")
    stats = integration_manager.manager.get_tool_performance_stats()
    for tool_name, tool_stats in stats.items():
        if tool_stats['total_executions'] > 0:
            print(f"  {tool_name}: {tool_stats['total_executions']} executions, "
                  f"{tool_stats.get('success_rate', 0):.1%} success rate")


def create_integration_documentation(setup_result, agent_configs):
    """Create documentation for the integration."""
    doc_path = Path(__file__).parent / 'INTEGRATION_GUIDE.md'
    
    integration_manager = setup_result['integration_manager']
    report = setup_result['report']
    
    doc_content = f"""# Microagent-Tools Integration Guide

## Overview

This guide describes the integration between the comprehensive tools system and microagents.

## Integration Status

- **Total Tools**: {report['tools_system']['total_tools']}
- **Tool Categories**: {report['tools_system']['categories']}
- **Registry Path**: {setup_result['registry_path']}
- **Configured Agents**: {len(agent_configs)}

## Available Tools

"""
    
    # Add tool categories
    for category, info in report['tools_system'].items():
        if isinstance(info, dict) and 'tools' in info:
            doc_content += f"### {category.replace('_', ' ').title()}\n"
            doc_content += f"{info['description']}\n"
            for tool in info['tools']:
                doc_content += f"- `{tool}`\n"
            doc_content += "\n"
    
    # Add agent configurations
    doc_content += "## Configured Agents\n\n"
    for agent_id, agent_info in agent_configs.items():
        doc_content += f"### {agent_id}\n"
        doc_content += f"- **Type**: {agent_info['type']}\n"
        doc_content += f"- **Available Tools**: {len(agent_info['available_tools'])}\n"
        doc_content += f"- **Tools**: {', '.join(agent_info['available_tools'])}\n"
        
        permissions = agent_info['config']['permissions']
        doc_content += f"- **Max Concurrent**: {permissions['max_concurrent_executions']}\n"
        doc_content += f"- **Timeout**: {permissions['timeout_seconds']}s\n"
        doc_content += f"- **Rate Limit**: {permissions['rate_limit_rpm']} req/min\n\n"
    
    # Add usage examples
    doc_content += """## Usage Examples

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
"""
    
    # Write documentation
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    
    print(f"✓ Integration documentation created: {doc_path}")
    return doc_path


async def main():
    """Main integration setup function."""
    parser = argparse.ArgumentParser(description='Setup microagent-tools integration')
    parser.add_argument('--registry-path', help='Path to microagent registry file')
    parser.add_argument('--config-agents', action='store_true', 
                       help='Configure example agents')
    parser.add_argument('--demo', action='store_true', 
                       help='Run integration demo')
    parser.add_argument('--docs', action='store_true',
                       help='Generate integration documentation')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        # No arguments, run complete setup
        args.config_agents = True
        args.demo = True
        args.docs = True
    
    print("="*60)
    print("MICROAGENT-TOOLS INTEGRATION SETUP")
    print("="*60)
    
    # Set up basic integration
    setup_result = setup_integration(args.registry_path)
    integration_manager = setup_result['integration_manager']
    
    agent_configs = {}
    
    # Configure example agents if requested
    if args.config_agents:
        agent_configs = configure_example_agents(integration_manager)
    
    # Run demo if requested
    if args.demo and agent_configs:
        await run_integration_demo(integration_manager, agent_configs)
    
    # Generate documentation if requested
    if args.docs:
        create_integration_documentation(setup_result, agent_configs)
    
    print("\n" + "="*60)
    print("INTEGRATION SETUP COMPLETED")
    print("="*60)
    print("The tools system is now integrated with microagents!")
    print(f"Registry file: {setup_result['registry_path']}")
    print(f"Total tools available: {setup_result['report']['tools_system']['total_tools']}")
    if agent_configs:
        print(f"Configured agents: {len(agent_configs)}")
    
    print("\nNext steps:")
    print("1. Review the generated documentation")
    print("2. Configure your specific microagents")
    print("3. Start using tools in your microagent workflows")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
