# Formal Action Tokenizer Documentation

## Overview

The Formal Action Tokenizer provides a mathematically rigorous foundation for converting natural language into structured action tokens. Built on formal action algebra with stable ABI management, type safety, and comprehensive evaluation capabilities.

## Quick Start

```python
from tokenizer import tokenize, create_token, ActionType

# Basic tokenization
tokens = tokenize("click the submit button")
print(f"Generated {len(tokens)} tokens")

# Create token directly
token = create_token("WAIT", {"duration": 5.0})
print(f"Created: {token.name} with args {token.args}")
```

## Core Concepts

### Action Algebra

Actions are defined by the formal algebra:
```
A = ⟨id, name, type, args, caps, meta⟩
```

Where:
- **id**: Stable 32-bit ABI identifier (never reused)
- **name**: Canonical action name (e.g., "CLICK", "TYPE")
- **type**: Classification (ACTION, CONTROL, SENSOR, EVENT)
- **args**: Typed parameters validated against schema
- **caps**: Required capabilities for execution
- **meta**: Additional metadata and context

### Tokenization Process

The tokenizer implements the mapping function:
```
f_θ: X → A₁:T
```

Where natural language input X is converted to a sequence of action tokens A₁:T through:

1. **Intent Parsing**: Extract structured intents from natural language
2. **Action Mapping**: Map intents to registered actions
3. **Parameter Resolution**: Resolve arguments and validate types
4. **Capability Filtering**: Filter based on available capabilities
5. **Token Creation**: Generate validated action tokens

## API Reference

### Core Classes

#### ActionTokenizer

Main tokenizer class implementing the formal tokenization function.

```python
tokenizer = ActionTokenizer(registry=None)

# Primary methods
tokens = tokenizer.tokenize(
    input_text="click submit button",
    available_capabilities={"dom.query", "input"},
    context={"page_url": "https://example.com"}
)

# Batch processing
batch_results = tokenizer.batch_tokenize([
    "click button",
    "type hello", 
    "wait 5 seconds"
])

# Statistics
stats = tokenizer.get_statistics()
```

#### ActionRegistry

Registry for action definitions with stable ABI management.

```python
registry = ActionRegistry()

# Register custom action
action_def = ActionDefinition(
    id=registry.next_id,
    name="SCROLL",
    action_type=ActionType.ACTION,
    params=[
        ParamSchema("direction", ParamType.ENUM, 
                   enum_values=["up", "down"]),
        ParamSchema("amount", ParamType.INT, default=1)
    ],
    capabilities={"dom.scroll"}
)
registry.register_action(action_def)

# Create tokens
token = registry.create_token("SCROLL", {
    "direction": "down", 
    "amount": 3
})

# ABI management
abi_data = registry.export_abi()
registry.import_abi(abi_data)
```

#### ActionToken

Individual action token with formal structure.

```python
token = ActionToken(
    id=1,
    name="CLICK",
    action_type=ActionType.ACTION,
    args={"target": "#submit-btn"},
    capabilities={"dom.query", "input"}
)

# Serialization
json_data = token.to_dict()
binary_data = token.to_binary()

# Validation
is_valid = token.validate_capabilities(available_caps)
duration = token.estimate_duration()
```

### Built-in Actions

| ID | Name | Type | Parameters | Capabilities |
|----|------|------|------------|--------------|
| 1 | CLICK | ACTION | target, button?, count? | dom.query, input |
| 2 | TYPE | ACTION | target, text, enter? | dom.query, input |
| 3 | WAIT | CONTROL | duration | timing |
| 4 | NAVIGATE | ACTION | url | net.fetch |
| 5 | SET | ACTION | key, value, scope? | storage |
| 6 | ASSERT | CONTROL | predicate, timeout? | dom.query |
| 7 | CAPTURE_IMG | SENSOR | region?, label? | camera, screen |
| 8 | LOOP | CONTROL | times | control |

### Parameter Types

- **STRING**: Text values
- **INT**: Integer numbers  
- **FLOAT**: Floating point numbers
- **BOOL**: True/false values
- **SELECTOR**: CSS/XPath selectors
- **URL**: Web addresses
- **DURATION**: Time durations in seconds
- **ENUM**: Enumerated values from fixed set
- **BOX**: Bounding boxes {x, y, width, height}
- **KVKEY/KVVALUE**: Key-value pairs
- **PREDICATE**: Boolean predicates
- **LIST**: Arrays of values
- **OBJECT**: Complex objects

## Performance & Testing

The formal tokenizer achieves:
- **Latency**: ~0.04ms mean response time
- **Throughput**: 25,000+ requests/second  
- **Accuracy**: 80% on basic test suite
- **Serialization**: CBOR ~60% smaller than JSON

Run the demonstration:
```bash
cd tokenizer && python demo.py
```

## Integration

This tokenizer integrates with:
- **Multimodal Context Router**: Action token routing
- **Microagents Registry**: Context-aware suggestions
- **Hardware Middleware**: Capability-based execution
- **Evaluation Framework**: Comprehensive quality metrics

Ready for tools integration phase with Tavily and other external APIs.
