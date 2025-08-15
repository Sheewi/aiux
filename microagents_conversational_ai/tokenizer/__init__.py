"""
Formal Action Tokenizer Package
Mathematical foundation for action tokenization with stable ABI and evaluation.
"""

from .action_tokenizer import (
    # Core classes
    ActionTokenizer,
    ActionRegistry, 
    ActionToken,
    ActionDefinition,
    ParamSchema,
    
    # Enums
    ActionType,
    ParamType,
    
    # Global instances
    default_registry,
    default_tokenizer,
    
    # Convenience functions
    tokenize,
    create_token,
    register_action,
    get_action
)

from .evaluation import (
    # Evaluation classes
    TokenizerEvaluator,
    EvaluationSuite,
    EvaluationResult,
    TestCase,
    
    # Metrics
    EvaluationMetric,
    
    # Test suites
    create_basic_test_suite,
    create_complex_test_suite,
    create_performance_test_suite,
    
    # Convenience functions
    evaluate_tokenizer,
    benchmark_tokenizer
)

__version__ = "1.0.0"
__author__ = "Microagents Conversational AI"
__description__ = "Formal mathematical action tokenizer with stable ABI"

# Package metadata
__all__ = [
    # Core tokenizer
    "ActionTokenizer",
    "ActionRegistry",
    "ActionToken", 
    "ActionDefinition",
    "ParamSchema",
    "ActionType",
    "ParamType",
    
    # Global instances
    "default_registry",
    "default_tokenizer",
    
    # Convenience functions
    "tokenize",
    "create_token", 
    "register_action",
    "get_action",
    
    # Evaluation framework
    "TokenizerEvaluator",
    "EvaluationSuite",
    "EvaluationResult",
    "TestCase",
    "EvaluationMetric",
    
    # Test suites
    "create_basic_test_suite",
    "create_complex_test_suite", 
    "create_performance_test_suite",
    
    # Evaluation functions
    "evaluate_tokenizer",
    "benchmark_tokenizer"
]

# Validate installation
def _check_installation():
    """Check if the tokenizer is properly installed."""
    try:
        # Test basic functionality
        test_tokens = tokenize("click button")
        if not test_tokens:
            raise RuntimeError("Tokenizer not functioning correctly")
        
        # Test registry
        action = get_action("CLICK")
        if not action:
            raise RuntimeError("Registry not properly initialized")
            
        return True
    except Exception as e:
        print(f"Warning: Tokenizer installation check failed: {e}")
        return False

# Run installation check on import
_installation_ok = _check_installation()

if not _installation_ok:
    print("⚠️  Action tokenizer may not be functioning correctly")
else:
    print("✅ Formal action tokenizer ready")
