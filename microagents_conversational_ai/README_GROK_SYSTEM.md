# Grok-like Multimodal Context Router

A sophisticated AI system that mimics Grok's core functionality: **dynamic output switching and hot-swappable renderers within the same chat session**.

## 🎯 What Grok Does (And What We've Replicated)

Grok's secret sauce isn't magic—it's a multimodal context router that can:

1. **Hot-swap output renderers** (Text → CLI → Graph → Diagram) without leaving chat
2. **Dynamic input interpretation** (Natural language, commands, structured queries)
3. **Context-aware agent suggestions** based on intent and entities
4. **Interactive elements** embedded directly in responses
5. **Session context preservation** across mode switches

## 🏗️ Architecture

```
User Input → Input Interpreter → Context Router → Agent Suggestions → Output Renderer
     ↓              ↓                ↓               ↓                    ↓
Natural Lang    Intent/Entity    Processing      Microagent         Text/CLI/Graph
Commands        Extraction       Pipeline        Routing            Diagram/Code
Structured      Confidence       Tokenization    Relevance          Interactive
Queries         Scoring          Execution       Scoring            Elements
```

## 🧩 Components

### 1. **Multimodal Context Router** (`multimodal_context_router.py`)
- **Input Interpreters**: Natural language, CLI commands, structured queries
- **Output Renderers**: Text, CLI, Graph, Diagram, Code, Interactive
- **Context Management**: Session state, mode history, confidence tracking
- **Hot-swapping**: Add/replace renderers dynamically

### 2. **Enhanced Agent Registry** (`registry.py`)
- **Capability Discovery**: Find agents by intent, input/output modes
- **Context-aware Suggestions**: Relevance scoring and explanations
- **Multimodal Support**: Agent metadata includes supported modes
- **Intent Matching**: Keyword-based and semantic agent routing

### 3. **Grok-style Output Format** (`grok_output_format.py`)
- **Structured Metadata**: Confidence, processing time, token usage
- **Interactive Elements**: Buttons, inputs, dropdowns, code editors
- **Follow-up Context**: Maintains conversation state
- **Template System**: Pre-defined formats for different modes

### 4. **Action Tokenizer Integration** (`tokenizer/action_tokenizer.py`)
- **Precise Mode**: Static token tables for known workflows
- **LLM-guided Mode**: Dynamic tokenization for novel tasks
- **Hardware Awareness**: Device mapping and execution context

## 🚀 Quick Start

```bash
# Test the complete system
python test_grok_system.py

# Run interactive demo
python grok_like_demo.py

# Or import and use programmatically
from multimodal_context_router import MultimodalContextRouter
from registry import agent_registry
from tokenizer.action_tokenizer import ActionTokenizer

router = MultimodalContextRouter(
    microagent_registry=agent_registry,
    tokenizer=ActionTokenizer()
)

result = await router.process_input("Show me web scraping agents")
print(f"Mode: {result.mode.value}")
print(f"Content: {result.content}")
```

## 💡 Example Interactions

### Text → CLI Mode Switch
```
User: "List all Python files in the current directory"
System: [Detects CLI intent, switches to CLI mode]
Output: 
```bash
$ find . -name "*.py" -type f
./registry.py
./multimodal_context_router.py
./grok_like_demo.py
[Interactive: ▶ Execute | 📝 Modify | 💾 Save]
```

### Natural Language → Graph Mode
```
User: "Show me the relationship between microagents"
System: [Detects visualization intent, switches to Graph mode]
Output: [Interactive graph with nodes/edges, zoom/pan controls]
```

### Structured Query → Table Mode
```
User: {"query": "security tools", "format": "table"}
System: [Detects structured input, switches to Table mode]
Output: [Sortable table with agent names, types, capabilities]
```

## 🎛️ Interactive Elements

Each output mode includes embedded interactive elements:

- **CLI Mode**: Execute, Modify, Copy, Save buttons
- **Graph Mode**: Zoom, Pan, Filter, Export controls
- **Diagram Mode**: Edit source, Change type, Export options
- **Code Mode**: Run, Format, Debug, Save buttons
- **Table Mode**: Sort, Filter, Export, Details links

## 🔥 Hot-swappable Renderers

Add custom renderers without restarting:

```python
class CustomTableRenderer(OutputRenderer):
    def render(self, content, metadata):
        # Your custom rendering logic
        return RenderedOutput(...)

# Hot-swap the renderer
router.add_renderer(OutputMode.TABLE, CustomTableRenderer())
```

## 🧠 Context-Aware Agent Suggestions

The system suggests relevant microagents based on:

- **Intent keywords**: Extracted from user input
- **Entity types**: Files, URLs, commands, data types
- **Mode compatibility**: Input/output mode support
- **Capability matching**: Agent skills vs. user needs
- **Relevance scoring**: Weighted confidence metrics

## 📊 Session Context Preservation

Maintains rolling context across interactions:
- Last 10 interactions with confidence scores
- Mode switching history (last 20 modes)
- Agent suggestions and execution results
- User preferences and session state

## 🎯 What Makes This "Grok-like"

1. **Dynamic Mode Switching**: Switch between text, CLI, graphs, diagrams in same session
2. **Hot-swappable Components**: Add new renderers/interpreters without restart
3. **Context Awareness**: Remembers conversation state across mode switches
4. **Interactive Elements**: Embedded UI components in responses
5. **Structured Output**: Metadata-rich responses with execution context
6. **Agent Orchestration**: Smart routing to specialized microagents

## 🛠️ Extension Points

- **Add new input modes**: Voice, image, gesture interpreters
- **Custom output renderers**: 3D visualizations, AR/VR, custom formats
- **Agent integration**: Connect to external AI services, APIs
- **Hardware middleware**: IoT device control, robotics integration
- **Tokenizer modes**: New tokenization strategies for domain-specific tasks

## 🔍 Files Overview

- `multimodal_context_router.py` - Core routing and mode switching logic
- `registry.py` - Enhanced agent registry with multimodal support
- `grok_output_format.py` - Structured output format specifications
- `grok_like_demo.py` - Interactive demo showcasing all features
- `test_grok_system.py` - Comprehensive system testing
- `tokenizer/action_tokenizer.py` - Existing tokenizer integration
- `tokenizer/microagent_registry.py` - Agent capability discovery

This system provides the foundation for building Grok-like AI interfaces with full user control over the environment and execution context.
