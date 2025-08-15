#!/usr/bin/env python3
"""
Formal Action Tokenizer Demo
Demonstrates the capabilities of the formal mathematical tokenizer system.
"""

import json
import time
from typing import Dict, Any, List

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from action_tokenizer import (
    ActionTokenizer, ActionRegistry, ActionDefinition, ActionType, 
    ParamSchema, ParamType, tokenize, create_token, register_action
)

# Import evaluation with proper module handling
try:
    from evaluation import (
        TokenizerEvaluator, create_basic_test_suite, create_complex_test_suite,
        evaluate_tokenizer, benchmark_tokenizer
    )
except ImportError:
    # Create minimal evaluation stubs for demo
    print("Note: Full evaluation framework not available, using basic demo")
    class TokenizerEvaluator:
        def __init__(self, tokenizer): pass
    def evaluate_tokenizer(tokenizer, suite_name): return {}
    def benchmark_tokenizer(tokenizer): return {}
    def create_basic_test_suite(): return None
    def create_complex_test_suite(): return None

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def demo_basic_tokenization():
    """Demonstrate basic tokenization functionality."""
    print_section("BASIC TOKENIZATION DEMO")
    
    test_inputs = [
        "click the submit button",
        'type "hello world"',
        "wait for 3 seconds",
        "navigate to https://example.com",
        "set username to alice"
    ]
    
    print("Testing natural language to action token conversion:")
    print()
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"{i}. Input: '{input_text}'")
        
        # Tokenize
        tokens = tokenize(input_text)
        
        if tokens:
            for j, token in enumerate(tokens):
                print(f"   Token {j+1}:")
                print(f"     Name: {token.name}")
                print(f"     Type: {token.action_type.name}")
                print(f"     Args: {token.args}")
                print(f"     ID: {token.id}")
                print(f"     Capabilities: {token.capabilities}")
        else:
            print("   No tokens generated")
        print()

def demo_registry_management():
    """Demonstrate action registry management."""
    print_section("REGISTRY MANAGEMENT DEMO")
    
    registry = ActionRegistry()
    
    print_subsection("Built-in Actions")
    actions = registry.list_actions()
    print(f"Registry contains {len(actions)} actions:")
    for action in actions:
        print(f"  {action.id:2}: {action.name:12} ({action.action_type.name})")
    
    print_subsection("Custom Action Registration")
    
    # Register a custom action
    custom_action = ActionDefinition(
        id=registry.next_id,
        name="SCROLL",
        action_type=ActionType.ACTION,
        params=[
            ParamSchema("direction", ParamType.ENUM, required=True,
                       enum_values=["up", "down", "left", "right"]),
            ParamSchema("amount", ParamType.INT, required=False, default=1,
                       constraints={"min": 1, "max": 10})
        ],
        capabilities={"dom.scroll"},
        description="Scroll in a direction"
    )
    
    success = registry.register_action(custom_action)
    print(f"Custom action registration: {'Success' if success else 'Failed'}")
    
    if success:
        # Test the custom action
        token = registry.create_token("SCROLL", {"direction": "down", "amount": 3})
        if token:
            print(f"Created token: {token.name} with args {token.args}")
    
    print_subsection("ABI Export/Import")
    
    # Export ABI
    abi_data = registry.export_abi()
    print(f"Exported ABI with {len(abi_data['actions'])} actions")
    print(f"Schema version: {abi_data['schema_version']}")
    print(f"Next ID: {abi_data['next_id']}")

def demo_advanced_features():
    """Demonstrate advanced tokenizer features."""
    print_section("ADVANCED FEATURES DEMO")
    
    tokenizer = ActionTokenizer()
    
    print_subsection("Capability Filtering")
    
    # Test with limited capabilities
    available_caps = {"dom.query", "input"}  # No network capabilities
    
    test_input = "navigate to google.com"
    tokens_no_filter = tokenizer.tokenize(test_input)
    tokens_filtered = tokenizer.tokenize(test_input, available_caps)
    
    print(f"Input: '{test_input}'")
    print(f"Without capability filter: {len(tokens_no_filter)} tokens")
    print(f"With capability filter: {len(tokens_filtered)} tokens")
    
    if tokens_no_filter and not tokens_filtered:
        required_caps = tokens_no_filter[0].capabilities
        missing_caps = required_caps - available_caps
        print(f"Filtered out due to missing capabilities: {missing_caps}")
    
    print_subsection("Token Serialization")
    
    if tokens_no_filter:
        token = tokens_no_filter[0]
        
        # Test JSON serialization
        json_data = json.dumps(token.to_dict(), indent=2)
        print("JSON serialization:")
        print(json_data[:200] + "..." if len(json_data) > 200 else json_data)
        
        # Test binary serialization
        binary_data = token.to_binary()
        print(f"\nBinary serialization: {len(binary_data)} bytes")
        
        # Test round-trip
        from action_tokenizer import ActionToken
        restored_token = ActionToken.from_binary(binary_data)
        print(f"Round-trip successful: {token.name == restored_token.name}")
    
    print_subsection("Batch Processing")
    
    batch_inputs = [
        "click submit",
        "type hello",
        "wait 1 second",
        "go to example.com"
    ]
    
    start_time = time.time()
    batch_results = tokenizer.batch_tokenize(batch_inputs)
    batch_time = time.time() - start_time
    
    print(f"Processed {len(batch_inputs)} inputs in {batch_time:.3f}s")
    print(f"Average time per input: {batch_time/len(batch_inputs):.3f}s")
    
    for i, (input_text, tokens) in enumerate(zip(batch_inputs, batch_results)):
        print(f"  {i+1}. '{input_text}' → {len(tokens)} tokens")

def demo_evaluation_framework():
    """Demonstrate the evaluation framework."""
    print_section("EVALUATION FRAMEWORK DEMO")
    
    tokenizer = ActionTokenizer()
    
    print_subsection("Basic Test Suite")
    
    # Run basic evaluation
    results = evaluate_tokenizer(tokenizer, "basic")
    
    print("Evaluation Results:")
    for metric_name, result in results.items():
        print(f"  {metric_name}: {result.value:.3f}")
        if result.details:
            for key, value in result.details.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.3f}")
    
    print_subsection("Performance Benchmark")
    
    # Run performance test
    perf_results = evaluate_tokenizer(tokenizer, "performance")
    
    if "latency" in perf_results:
        latency_result = perf_results["latency"]
        details = latency_result.details
        print("Latency Statistics:")
        print(f"  Mean: {details['mean_ms']:.2f}ms")
        print(f"  Median: {details['median_ms']:.2f}ms")
        print(f"  P95: {details['p95_ms']:.2f}ms")
        print(f"  P99: {details['p99_ms']:.2f}ms")
    
    if "throughput" in perf_results:
        throughput_result = perf_results["throughput"]
        details = throughput_result.details
        print(f"\nThroughput: {details['requests_per_second']:.1f} req/s")

def demo_error_handling():
    """Demonstrate error handling and validation."""
    print_section("ERROR HANDLING DEMO")
    
    registry = ActionRegistry()
    
    print_subsection("Invalid Token Creation")
    
    # Try to create invalid tokens
    test_cases = [
        ("CLICK", {}),  # Missing required parameter
        ("CLICK", {"target": "#btn", "count": -1}),  # Invalid parameter value
        ("UNKNOWN_ACTION", {"param": "value"}),  # Unknown action
        ("WAIT", {"duration": "invalid"}),  # Wrong parameter type
    ]
    
    for action_name, args in test_cases:
        print(f"Attempting to create {action_name} with args {args}")
        token = registry.create_token(action_name, args)
        if token:
            print(f"  ✓ Created successfully")
        else:
            print(f"  ✗ Failed (as expected)")
    
    print_subsection("Malformed Input Handling")
    
    tokenizer = ActionTokenizer()
    malformed_inputs = [
        "",  # Empty input
        "asdfghjkl",  # Random text
        "click",  # Incomplete command
        "click the button with id",  # Incomplete selector
    ]
    
    for input_text in malformed_inputs:
        tokens = tokenizer.tokenize(input_text)
        print(f"'{input_text}' → {len(tokens)} tokens")

def main():
    """Run the complete demonstration."""
    print("🚀 FORMAL ACTION TOKENIZER DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the mathematical foundation and")
    print("capabilities of the formal action tokenizer system.")
    
    try:
        demo_basic_tokenization()
        demo_registry_management()
        demo_advanced_features()
        demo_evaluation_framework()
        demo_error_handling()
        
        print_section("DEMONSTRATION COMPLETE")
        print("✅ All demos completed successfully!")
        print("\nThe formal action tokenizer provides:")
        print("  • Stable ABI with unique action IDs")
        print("  • Type-safe parameter validation")
        print("  • Capability-based filtering")
        print("  • Binary serialization with CBOR/JSON")
        print("  • Comprehensive evaluation framework")
        print("  • Robust error handling")
        print("\nReady for integration with tools and microagents!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
