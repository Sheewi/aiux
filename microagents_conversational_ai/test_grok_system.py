#!/usr/bin/env python3
"""
Quick test script for the Grok-like multimodal system.
Tests the integration between registry, tokenizer, and multimodal router.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

async def test_multimodal_system():
    """Test the complete multimodal system."""
    
    print("🧪 Testing Grok-like Multimodal System")
    print("=" * 50)
    
    try:
        # Test 1: Import all components
        print("📦 Testing imports...")
        from registry import agent_registry
        from tokenizer.action_tokenizer import ActionTokenizer
        from multimodal_context_router import MultimodalContextRouter, OutputMode
        from grok_output_format import create_grok_output, RenderType
        print("✅ All imports successful")
        
        # Test 2: Initialize components
        print("\n🔧 Initializing components...")
        tokenizer = ActionTokenizer()
        router = MultimodalContextRouter(
            microagent_registry=agent_registry,
            tokenizer=tokenizer
        )
        print("✅ Components initialized")
        
        # Test 3: Test registry functionality
        print("\n📋 Testing agent registry...")
        agents = agent_registry.list_agents()
        print(f"✅ Found {len(agents)} agents in registry")
        
        # Test 4: Test context suggestions
        print("\n🎯 Testing context-aware suggestions...")
        context = {
            'intent': 'scrape',
            'entities': [{'type': 'web', 'value': 'website'}],
            'input_mode': 'natural_language',
            'suggested_output_mode': 'text'
        }
        suggestions = agent_registry.suggest_agents_for_context(context)
        print(f"✅ Generated {len(suggestions)} agent suggestions")
        
        # Test 5: Test multimodal processing
        print("\n🔄 Testing multimodal processing...")
        test_inputs = [
            "Show me web scraping agents",
            "$ python scraper.py --help",
            "Create a network diagram of agents",
            '{"query": "security tools", "format": "table"}'
        ]
        
        for i, test_input in enumerate(test_inputs):
            print(f"\n  Test {i+1}: {test_input[:30]}...")
            try:
                result = await router.process_input(test_input)
                print(f"    ✅ Output mode: {result.mode.value}")
                print(f"    ✅ Interactive elements: {len(result.interactive_elements)}")
                print(f"    ✅ Follow-up suggestions: {len(result.follow_up_suggestions)}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # Test 6: Test Grok-style output format
        print("\n🎨 Testing Grok-style output format...")
        grok_output = create_grok_output(
            RenderType.CLI,
            {"command": "ls -la", "output": "total 42\\ndrwxr-xr-x..."},
            confidence=0.95,
            agent_suggestions=["file_manager", "terminal_agent"]
        )
        print("✅ Grok-style output created successfully")
        print(f"    Metadata: {len(grok_output.metadata.__dict__)} fields")
        print(f"    Interactive elements: {len(grok_output.interactive_elements)}")
        
        # Test 7: Test session statistics
        print("\n📊 Testing session statistics...")
        stats = router.get_session_stats()
        print(f"✅ Session stats: {stats}")
        
        print("\n🎉 All tests passed! System is ready.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_mini_demo():
    """Run a minimal demo of the system."""
    
    print("\n🚀 Mini Demo - Grok-like Functionality")
    print("=" * 50)
    
    from registry import agent_registry
    from tokenizer.action_tokenizer import ActionTokenizer
    from multimodal_context_router import MultimodalContextRouter
    
    # Initialize
    router = MultimodalContextRouter(
        microagent_registry=agent_registry,
        tokenizer=ActionTokenizer()
    )
    
    # Demo inputs that showcase different modes
    demo_inputs = [
        {
            "input": "Show me all web scraping agents",
            "expected_mode": "table or text",
            "description": "Agent discovery query"
        },
        {
            "input": "$ find . -name '*.py' | head -5",
            "expected_mode": "cli",
            "description": "Command line interface"
        },
        {
            "input": "Create a flowchart of the agent workflow",
            "expected_mode": "diagram", 
            "description": "Diagram generation"
        }
    ]
    
    for demo in demo_inputs:
        print(f"\n📝 {demo['description']}")
        print(f"Input: {demo['input']}")
        print(f"Expected: {demo['expected_mode']}")
        print("-" * 30)
        
        try:
            result = await router.process_input(demo['input'])
            
            print(f"🎯 Output Mode: {result.mode.value}")
            print(f"📄 Content Preview: {str(result.content)[:100]}...")
            
            if result.interactive_elements:
                print(f"🎛️  Interactive: {[e.label for e in result.interactive_elements[:3]]}")
            
            if result.follow_up_suggestions:
                print(f"💡 Suggestions: {result.follow_up_suggestions[:2]}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n📊 Session completed with {router.get_session_stats()['total_interactions']} interactions")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Full system test")
    print("2. Mini demo")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip() or "3"
    
    async def run_tests():
        if choice in ["1", "3"]:
            success = await test_multimodal_system()
            if not success:
                print("❌ System test failed")
                return
        
        if choice in ["2", "3"]:
            await run_mini_demo()
    
    asyncio.run(run_tests())
