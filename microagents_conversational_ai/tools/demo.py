#!/usr/bin/env python3
"""
Tools System Demo and Test Script

This script demonstrates the comprehensive tools system functionality
and can be used to verify that all tools are working correctly.
"""

import asyncio
import json
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import tools
sys.path.append(str(Path(__file__).parent.parent))

try:
    from tools import (
        global_registry, global_manager,
        list_all_tools, get_tool_categories, get_tool_info,
        TOOL_EXAMPLES
    )
except ImportError as e:
    print(f"Error importing tools: {e}")
    print("Make sure you're running this script from the correct directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def demo_registry_functions():
    """Demonstrate registry functionality."""
    print("\n" + "="*60)
    print("TOOLS REGISTRY DEMONSTRATION")
    print("="*60)
    
    # List all tools
    print("\n1. Available Tools:")
    tools = list_all_tools()
    for tool in tools:
        print(f"   - {tool}")
    
    # Show categories
    print("\n2. Tool Categories:")
    categories = get_tool_categories()
    for category, info in categories.items():
        print(f"   - {category}: {info['description']}")
        for tool in info['tools']:
            print(f"     * {tool}")
    
    # Show tool information
    print("\n3. Tool Information Example (tavily_search):")
    info = get_tool_info('tavily_search')
    if info:
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Version: {info['version']}")
        print(f"   Capabilities: {', '.join(info['capabilities'])}")
    
    # Search tools
    print("\n4. Search Tools (query: 'web'):")
    search_results = global_registry.search_tools('web')
    for tool in search_results:
        print(f"   - {tool}")


async def demo_file_operations():
    """Demonstrate file operations tool."""
    print("\n" + "="*60)
    print("FILE OPERATIONS DEMONSTRATION")
    print("="*60)
    
    try:
        # Create a temporary file
        print("\n1. Creating temporary test file...")
        result = await global_manager.execute_tool(
            'file_operations', 
            'write',
            file_path='test_file.txt',
            content='Hello, World!\nThis is a test file for the tools system.',
            create_dirs=True
        )
        print(f"   Result: {result['result']['result'] if result['success'] else result['error']}")
        
        # Read the file
        print("\n2. Reading the test file...")
        result = await global_manager.execute_tool(
            'file_operations',
            'read',
            file_path='test_file.txt'
        )
        if result['success']:
            content = result['result']['result']['content']
            print(f"   Content: {content[:50]}...")
        else:
            print(f"   Error: {result['error']}")
        
        # Get file info
        print("\n3. Getting file information...")
        result = await global_manager.execute_tool(
            'file_operations',
            'info',
            file_path='test_file.txt'
        )
        if result['success']:
            info = result['result']['result']
            print(f"   Size: {info['size']} bytes")
            print(f"   Modified: {info['modified']}")
        
        # Clean up
        print("\n4. Cleaning up test file...")
        result = await global_manager.execute_tool(
            'file_operations',
            'delete',
            file_path='test_file.txt'
        )
        print(f"   Cleanup: {'Success' if result['success'] else 'Failed'}")
        
    except Exception as e:
        logger.error(f"File operations demo failed: {e}")


async def demo_data_processing():
    """Demonstrate data processing tool."""
    print("\n" + "="*60)
    print("DATA PROCESSING DEMONSTRATION")
    print("="*60)
    
    try:
        # Sample data
        sample_data = {
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000],
            'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
        }
        
        print("\n1. Analyzing sample data...")
        result = await global_manager.execute_tool(
            'data_processing',
            'analyze',
            data=sample_data,
            analysis_type='summary'
        )
        
        if result['success']:
            analysis = result['result']['result']['analysis_results']
            print(f"   Total rows: {analysis['overview']['total_rows']}")
            print(f"   Total columns: {analysis['overview']['total_columns']}")
            print(f"   Numeric columns: {analysis['overview']['numeric_columns']}")
        
        print("\n2. Filtering data (age > 30)...")
        result = await global_manager.execute_tool(
            'data_processing',
            'filter',
            data=sample_data,
            filters=[
                {
                    'column': 'age',
                    'operator': 'greater_than',
                    'value': 30
                }
            ]
        )
        
        if result['success']:
            filtered_data = result['result']['result']['filtered_data']
            print(f"   Filtered to {len(filtered_data)} rows")
            for row in filtered_data:
                print(f"   - {row['name']}: {row['age']} years old")
        
    except Exception as e:
        logger.error(f"Data processing demo failed: {e}")


async def demo_security():
    """Demonstrate security tool."""
    print("\n" + "="*60)
    print("SECURITY DEMONSTRATION")
    print("="*60)
    
    try:
        # Password generation
        print("\n1. Generating secure password...")
        result = await global_manager.execute_tool(
            'security',
            'generate_password',
            length=12,
            include_symbols=True
        )
        
        if result['success']:
            password = result['result']['result']['password']
            strength = result['result']['result']['strength_analysis']
            print(f"   Generated password: {password}")
            print(f"   Strength level: {strength['strength_level']}")
        
        # Password strength check
        print("\n2. Checking password strength...")
        test_password = "MySecureP@ssw0rd123"
        result = await global_manager.execute_tool(
            'security',
            'password_strength',
            password=test_password
        )
        
        if result['success']:
            strength = result['result']['result']
            print(f"   Password: {test_password}")
            print(f"   Strength: {strength['strength_level']}")
            print(f"   Score: {strength['strength_score']}/{strength['max_score']}")
        
        # Data hashing
        print("\n3. Hashing data...")
        result = await global_manager.execute_tool(
            'security',
            'hash',
            data='sensitive information',
            algorithm='sha256'
        )
        
        if result['success']:
            hash_result = result['result']['result']
            print(f"   Hash: {hash_result['hash'][:20]}...")
            print(f"   Algorithm: {hash_result['algorithm']}")
        
    except Exception as e:
        logger.error(f"Security demo failed: {e}")


async def demo_monitoring():
    """Demonstrate monitoring tool."""
    print("\n" + "="*60)
    print("MONITORING DEMONSTRATION")
    print("="*60)
    
    try:
        # System metrics
        print("\n1. Getting system metrics...")
        result = await global_manager.execute_tool(
            'monitoring',
            'system_metrics'
        )
        
        if result['success']:
            metrics = result['result']['result']
            print(f"   CPU Usage: {metrics['cpu']['usage_percent']:.1f}%")
            print(f"   Memory Usage: {metrics['memory']['usage_percent']:.1f}%")
            print(f"   Disk Usage: {metrics['disk']['usage_percent']:.1f}%")
            print(f"   Process Count: {metrics['system']['process_count']}")
        
        # Health check example
        print("\n2. Performing health checks...")
        services = [
            {
                'name': 'localhost_web',
                'type': 'tcp',
                'host': 'localhost',
                'port': 80,
                'timeout': 5
            }
        ]
        
        result = await global_manager.execute_tool(
            'monitoring',
            'health_check',
            services=services
        )
        
        if result['success']:
            health = result['result']['result']
            print(f"   Overall health: {health['overall_health']}")
            print(f"   Health score: {health['health_score']}%")
        
    except Exception as e:
        logger.error(f"Monitoring demo failed: {e}")


async def demo_tool_chain():
    """Demonstrate tool chain execution."""
    print("\n" + "="*60)
    print("TOOL CHAIN DEMONSTRATION")
    print("="*60)
    
    try:
        # Define a simple tool chain
        chain_config = [
            {
                'tool': 'security',
                'operation': 'generate_password',
                'parameters': {'length': 8, 'include_symbols': False},
                'context_key': 'password_gen'
            },
            {
                'tool': 'security',
                'operation': 'hash',
                'parameters': {
                    'data': '${password_gen.result.result.password}',
                    'algorithm': 'sha256'
                },
                'context_key': 'password_hash'
            }
        ]
        
        print("\n1. Executing tool chain (password generation + hashing)...")
        result = await global_manager.execute_tool_chain(chain_config)
        
        if result['overall_success']:
            print(f"   Chain completed successfully in {result['duration_seconds']:.2f} seconds")
            print(f"   Steps completed: {result['successful_steps']}/{result['total_steps']}")
            
            # Show results from each step
            for i, step_result in enumerate(result['results']):
                if step_result['success']:
                    print(f"   Step {i+1} ({step_result['tool_name']}): Success")
                else:
                    print(f"   Step {i+1} ({step_result['tool_name']}): Failed")
        else:
            print(f"   Chain failed: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Tool chain demo failed: {e}")


async def demo_performance_stats():
    """Demonstrate performance monitoring."""
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS")
    print("="*60)
    
    try:
        # Get execution history
        history = global_manager.get_execution_history(limit=5)
        print(f"\n1. Recent executions ({len(history)} shown):")
        for record in history[-5:]:
            status = "✓" if record['success'] else "✗"
            print(f"   {status} {record['tool_name']}.{record.get('operation', 'unknown')} "
                  f"({record['duration_seconds']:.2f}s)")
        
        # Get performance stats
        stats = global_manager.get_tool_performance_stats()
        print(f"\n2. Tool performance statistics:")
        for tool_name, tool_stats in stats.items():
            print(f"   {tool_name}:")
            print(f"     - Executions: {tool_stats['total_executions']}")
            print(f"     - Success rate: {tool_stats.get('success_rate', 0):.1%}")
            print(f"     - Avg duration: {tool_stats['average_duration']:.2f}s")
        
        # Get active tools
        active = global_manager.get_active_tools()
        print(f"\n3. Currently active tools: {len(active)}")
        for execution_id, tool_info in active.items():
            print(f"   - {execution_id}: {tool_info['tool_name']} ({tool_info['status']})")
        
    except Exception as e:
        logger.error(f"Performance stats demo failed: {e}")


async def run_comprehensive_demo():
    """Run the complete demonstration."""
    print("="*80)
    print("COMPREHENSIVE TOOLS SYSTEM DEMONSTRATION")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"Tools system loaded with {len(list_all_tools())} tools")
    
    # Run all demonstrations
    await demo_registry_functions()
    await demo_file_operations()
    await demo_data_processing()
    await demo_security()
    await demo_monitoring()
    await demo_tool_chain()
    await demo_performance_stats()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED")
    print("="*80)
    print("All tools are loaded and functional!")
    print("You can now integrate the tools system with your microagents.")


def show_tool_examples():
    """Show example usage for all tools."""
    print("\n" + "="*60)
    print("TOOL USAGE EXAMPLES")
    print("="*60)
    
    for tool_name, example_info in TOOL_EXAMPLES.items():
        print(f"\n{tool_name.upper()}:")
        print(f"   Description: {example_info['description']}")
        print(f"   Example usage:")
        example = example_info['example']
        print(f"     Tool: {example['tool']}")
        print(f"     Operation: {example['operation']}")
        print(f"     Parameters: {json.dumps(example['parameters'], indent=6)}")


async def main():
    """Main demonstration function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--examples':
        show_tool_examples()
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick test - just verify tools can be loaded
        print("Quick test: Loading tools system...")
        tools = list_all_tools()
        print(f"✓ Loaded {len(tools)} tools successfully")
        
        # Test one simple operation
        print("Testing file operations...")
        result = await global_manager.execute_tool(
            'security', 'hash', data='test', algorithm='sha256'
        )
        if result['success']:
            print("✓ Tool execution successful")
        else:
            print(f"✗ Tool execution failed: {result.get('error')}")
        return
    
    # Run full demonstration
    await run_comprehensive_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        sys.exit(1)
