"""
Grok-like Multimodal Context Router
Dynamic output renderer and input interpreter switching within chat sessions.
"""

import asyncio
import json
import logging
import re
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from grok_output_format import (
    GrokStyleOutput, 
    GrokStyleMetadata, 
    InteractiveElement, 
    RenderType,
    create_grok_output
)

class OutputMode(Enum):
    """Available output rendering modes."""
    TEXT = "text"
    CLI = "cli"
    GRAPH = "graph"
    DIAGRAM = "diagram"
    CODE = "code"
    INTERACTIVE = "interactive"
    VISUALIZATION = "visualization"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    TABLE = "table"
    TREE = "tree"

class InputMode(Enum):
    """Available input interpretation modes."""
    NATURAL_LANGUAGE = "natural_language"
    COMMAND_LINE = "command_line"
    CODE_BLOCK = "code_block"
    STRUCTURED_QUERY = "structured_query"
    VOICE = "voice"
    IMAGE = "image"
    FILE_UPLOAD = "file_upload"
    GESTURE = "gesture"

@dataclass
class ContextMetadata:
    """Metadata for context routing decisions."""
    output_mode: OutputMode
    input_mode: InputMode
    confidence: float
    reasoning: str
    requires_tools: List[str]
    estimated_complexity: str
    session_context: Dict[str, Any]

@dataclass
class RenderedOutput:
    """Container for rendered output with metadata."""
    content: Any
    mode: OutputMode
    metadata: Dict[str, Any]
    interactive_elements: List[Dict[str, Any]]
    follow_up_suggestions: List[str]

class OutputRenderer(ABC):
    """Abstract base class for output renderers."""
    
    @abstractmethod
    def render(self, content: Any, metadata: Dict[str, Any]) -> RenderedOutput:
        """Render content in the specific output mode."""
        pass
    
    @abstractmethod
    def can_handle(self, content_type: str, metadata: Dict[str, Any]) -> bool:
        """Check if this renderer can handle the given content."""
        pass

class InputInterpreter(ABC):
    """Abstract base class for input interpreters."""
    
    @abstractmethod
    def interpret(self, raw_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret raw input into structured format."""
        pass
    
    @abstractmethod
    def can_handle(self, input_type: str, context: Dict[str, Any]) -> bool:
        """Check if this interpreter can handle the given input."""
        pass

class TextRenderer(OutputRenderer):
    """Standard text output renderer."""
    
    def render(self, content: Any, metadata: Dict[str, Any]) -> RenderedOutput:
        if isinstance(content, str):
            text_content = content
        else:
            text_content = str(content)
        
        return RenderedOutput(
            content=text_content,
            mode=OutputMode.TEXT,
            metadata=metadata,
            interactive_elements=[],
            follow_up_suggestions=self._generate_follow_ups(content, metadata)
        )
    
    def can_handle(self, content_type: str, metadata: Dict[str, Any]) -> bool:
        return content_type in ["text", "string", "markdown"]
    
    def _generate_follow_ups(self, content: Any, metadata: Dict[str, Any]) -> List[str]:
        """Generate contextual follow-up suggestions."""
        return [
            "Would you like more details?",
            "Show this in a different format",
            "Execute this as a command"
        ]

class CLIRenderer(OutputRenderer):
    """Command-line interface output renderer."""
    
    def render(self, content: Any, metadata: Dict[str, Any]) -> RenderedOutput:
        if isinstance(content, dict) and 'command' in content:
            cli_content = f"$ {content['command']}\n{content.get('output', '')}"
        else:
            cli_content = f"$ {content}"
        
        return RenderedOutput(
            content=cli_content,
            mode=OutputMode.CLI,
            metadata=metadata,
            interactive_elements=[
                {
                    "type": "button",
                    "label": "Run Command",
                    "action": "execute_command",
                    "params": {"command": content}
                }
            ],
            follow_up_suggestions=[
                "Execute this command",
                "Modify parameters",
                "Show command help"
            ]
        )
    
    def can_handle(self, content_type: str, metadata: Dict[str, Any]) -> bool:
        return content_type in ["command", "cli", "terminal", "shell"]

class GraphRenderer(OutputRenderer):
    """Graph visualization output renderer."""
    
    def render(self, content: Any, metadata: Dict[str, Any]) -> RenderedOutput:
        # Convert content to graph format
        if isinstance(content, dict) and 'nodes' in content:
            graph_data = content
        else:
            # Try to extract graph structure from content
            graph_data = self._extract_graph_structure(content)
        
        return RenderedOutput(
            content=graph_data,
            mode=OutputMode.GRAPH,
            metadata=metadata,
            interactive_elements=[
                {
                    "type": "graph_viewer",
                    "config": {
                        "layout": "force-directed",
                        "zoom": True,
                        "pan": True
                    }
                }
            ],
            follow_up_suggestions=[
                "Expand node details",
                "Change layout",
                "Export as image"
            ]
        )
    
    def can_handle(self, content_type: str, metadata: Dict[str, Any]) -> bool:
        return content_type in ["graph", "network", "tree", "hierarchy", "relationship"]
    
    def _extract_graph_structure(self, content: Any) -> Dict[str, Any]:
        """Extract graph structure from various content types."""
        # Simplified extraction - in practice, this would be more sophisticated
        return {
            "nodes": [{"id": "node1", "label": "Content"}],
            "edges": []
        }

class DiagramRenderer(OutputRenderer):
    """Diagram output renderer (Mermaid, PlantUML, etc.)."""
    
    def render(self, content: Any, metadata: Dict[str, Any]) -> RenderedOutput:
        diagram_content = self._generate_diagram(content, metadata)
        
        return RenderedOutput(
            content=diagram_content,
            mode=OutputMode.DIAGRAM,
            metadata=metadata,
            interactive_elements=[
                {
                    "type": "diagram_editor",
                    "syntax": metadata.get("diagram_type", "mermaid")
                }
            ],
            follow_up_suggestions=[
                "Edit diagram",
                "Export as SVG",
                "Add annotations"
            ]
        )
    
    def can_handle(self, content_type: str, metadata: Dict[str, Any]) -> bool:
        return content_type in ["diagram", "flowchart", "sequence", "class_diagram", "uml"]
    
    def _generate_diagram(self, content: Any, metadata: Dict[str, Any]) -> str:
        """Generate diagram markup from content."""
        diagram_type = metadata.get("diagram_type", "mermaid")
        
        if diagram_type == "mermaid":
            return f"""
graph TD
    A[{content}] --> B[Process]
    B --> C[Output]
"""
        return str(content)

class NaturalLanguageInterpreter(InputInterpreter):
    """Natural language input interpreter."""
    
    def interpret(self, raw_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Analyze input for intent, entities, and required output mode
        intent = self._extract_intent(raw_input)
        entities = self._extract_entities(raw_input)
        output_mode = self._suggest_output_mode(raw_input, intent)
        
        return {
            "type": "natural_language",
            "intent": intent,
            "entities": entities,
            "suggested_output_mode": output_mode,
            "confidence": self._calculate_confidence(raw_input, intent),
            "processing_steps": self._plan_processing(intent, entities)
        }
    
    def can_handle(self, input_type: str, context: Dict[str, Any]) -> bool:
        return input_type == "text" or input_type == "natural_language"
    
    def _extract_intent(self, text: str) -> str:
        """Extract primary intent from natural language."""
        # Simplified intent detection
        if any(word in text.lower() for word in ["show", "display", "visualize"]):
            return "display"
        elif any(word in text.lower() for word in ["run", "execute", "command"]):
            return "execute"
        elif any(word in text.lower() for word in ["create", "generate", "build"]):
            return "create"
        elif any(word in text.lower() for word in ["analyze", "examine", "investigate"]):
            return "analyze"
        return "general"
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        # Simplified entity extraction
        entities = []
        
        # Look for file paths
        file_matches = re.findall(r'[./][\w/.-]+\.\w+', text)
        for match in file_matches:
            entities.append({"type": "file", "value": match})
        
        # Look for URLs
        url_matches = re.findall(r'https?://[\w.-]+', text)
        for match in url_matches:
            entities.append({"type": "url", "value": match})
        
        return entities
    
    def _suggest_output_mode(self, text: str, intent: str) -> OutputMode:
        """Suggest appropriate output mode based on input."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["graph", "network", "diagram"]):
            return OutputMode.GRAPH
        elif any(word in text_lower for word in ["command", "terminal", "shell"]):
            return OutputMode.CLI
        elif any(word in text_lower for word in ["code", "script", "function"]):
            return OutputMode.CODE
        elif any(word in text_lower for word in ["table", "data", "list"]):
            return OutputMode.TABLE
        else:
            return OutputMode.TEXT
    
    def _calculate_confidence(self, text: str, intent: str) -> float:
        """Calculate confidence in interpretation."""
        # Simplified confidence calculation
        return 0.8 if intent != "general" else 0.5
    
    def _plan_processing(self, intent: str, entities: List[Dict[str, Any]]) -> List[str]:
        """Plan processing steps based on intent and entities."""
        steps = []
        
        if intent == "execute" and any(e["type"] == "file" for e in entities):
            steps.append("validate_file_exists")
            steps.append("check_execution_permissions")
            steps.append("execute_file")
        elif intent == "display" and any(e["type"] == "url" for e in entities):
            steps.append("fetch_url_content")
            steps.append("render_content")
        else:
            steps.append("process_general_request")
        
        return steps

class MultimodalContextRouter:
    """
    Main router that orchestrates input interpretation and output rendering.
    Maintains session context and dynamically switches modes.
    """
    
    def __init__(self, microagent_registry=None, tokenizer=None):
        self.microagent_registry = microagent_registry
        self.tokenizer = tokenizer
        self.session_context = {}
        
        # Initialize renderers
        self.renderers = {
            OutputMode.TEXT: TextRenderer(),
            OutputMode.CLI: CLIRenderer(),
            OutputMode.GRAPH: GraphRenderer(),
            OutputMode.DIAGRAM: DiagramRenderer(),
        }
        
        # Initialize interpreters
        self.interpreters = {
            InputMode.NATURAL_LANGUAGE: NaturalLanguageInterpreter(),
        }
        
        # Mode history for context
        self.mode_history = []
    
    async def process_input(self, raw_input: str, input_mode: Optional[InputMode] = None) -> RenderedOutput:
        """
        Main processing pipeline: input -> interpretation -> routing -> rendering.
        """
        # 1. Determine input mode if not specified
        if input_mode is None:
            input_mode = self._detect_input_mode(raw_input)
        
        # 2. Interpret input
        interpreted = await self._interpret_input(raw_input, input_mode)
        
        # 3. Route to appropriate processor
        processing_result = await self._route_and_process(interpreted)
        
        # 4. Determine output mode
        output_mode = self._determine_output_mode(processing_result, interpreted)
        
        # 5. Render output
        rendered = await self._render_output(processing_result, output_mode)
        
        # 6. Update session context
        self._update_session_context(raw_input, interpreted, processing_result, rendered)
        
        return rendered
    
    def _detect_input_mode(self, raw_input: str) -> InputMode:
        """Detect the most appropriate input mode for raw input."""
        # Simple heuristics - in practice, this could use ML
        if raw_input.startswith(('$', '>', 'sudo ', 'cd ', 'ls ', 'cat ')):
            return InputMode.COMMAND_LINE
        elif raw_input.startswith(('```', 'def ', 'class ', 'import ', 'from ')):
            return InputMode.CODE_BLOCK
        elif raw_input.startswith(('{', '[')) and raw_input.endswith(('}', ']')):
            return InputMode.STRUCTURED_QUERY
        else:
            return InputMode.NATURAL_LANGUAGE
    
    async def _interpret_input(self, raw_input: str, input_mode: InputMode) -> Dict[str, Any]:
        """Interpret raw input using appropriate interpreter."""
        interpreter = self.interpreters.get(input_mode)
        if not interpreter:
            # Fallback to natural language
            interpreter = self.interpreters[InputMode.NATURAL_LANGUAGE]
        
        return interpreter.interpret(raw_input, self.session_context)
    
    async def _route_and_process(self, interpreted: Dict[str, Any]) -> Dict[str, Any]:
        """Route interpreted input to appropriate processing pipeline."""
        intent = interpreted.get("intent", "general")
        entities = interpreted.get("entities", [])
        
        # Determine if we need to use microagents
        if self.microagent_registry and intent in ["execute", "analyze", "create"]:
            return await self._process_with_microagents(interpreted)
        
        # Use tokenizer if available
        if self.tokenizer and intent == "execute":
            return await self._process_with_tokenizer(interpreted)
        
        # Default processing
        return {
            "type": "simple_response",
            "content": f"Processed intent: {intent}",
            "metadata": interpreted
        }
    
    async def _process_with_microagents(self, interpreted: Dict[str, Any]) -> Dict[str, Any]:
        """Process using microagent ecosystem."""
        intent = interpreted.get("intent")
        entities = interpreted.get("entities", [])
        
        # Find suitable microagents
        suitable_agents = []
        if self.microagent_registry:
            all_agents = self.microagent_registry.list_agents()
            # Simple matching - in practice, this would be more sophisticated
            for name, agent_class in all_agents.items():
                if intent.lower() in name.lower():
                    suitable_agents.append(name)
        
        return {
            "type": "microagent_response",
            "content": f"Found {len(suitable_agents)} suitable agents: {', '.join(suitable_agents)}",
            "suggested_agents": suitable_agents,
            "metadata": interpreted
        }
    
    async def _process_with_tokenizer(self, interpreted: Dict[str, Any]) -> Dict[str, Any]:
        """Process using action tokenizer."""
        if not self.tokenizer:
            return {"type": "error", "content": "No tokenizer available"}
        
        # Convert interpreted input to tokenizer format
        # This would integrate with the existing action_tokenizer.py
        return {
            "type": "tokenized_response",
            "content": "Tokenized for execution",
            "tokens": [],  # Would be actual tokens from tokenizer
            "metadata": interpreted
        }
    
    def _determine_output_mode(self, processing_result: Dict[str, Any], interpreted: Dict[str, Any]) -> OutputMode:
        """Determine the best output mode for the processing result."""
        # Check if input suggested a specific mode
        suggested_mode = interpreted.get("suggested_output_mode")
        if suggested_mode:
            return suggested_mode
        
        # Determine based on processing result type
        result_type = processing_result.get("type", "simple_response")
        
        if result_type == "microagent_response" and processing_result.get("suggested_agents"):
            return OutputMode.TABLE
        elif "command" in str(processing_result.get("content", "")).lower():
            return OutputMode.CLI
        elif "agents" in str(processing_result.get("content", "")).lower():
            return OutputMode.GRAPH
        else:
            return OutputMode.TEXT
    
    async def _render_output(self, processing_result: Dict[str, Any], output_mode: OutputMode) -> RenderedOutput:
        """Render processing result using appropriate renderer."""
        renderer = self.renderers.get(output_mode)
        if not renderer:
            # Fallback to text renderer
            renderer = self.renderers[OutputMode.TEXT]
        
        content = processing_result.get("content", "No content")
        metadata = processing_result.get("metadata", {})
        
        return renderer.render(content, metadata)
    
    def _update_session_context(self, raw_input: str, interpreted: Dict[str, Any], 
                              processing_result: Dict[str, Any], rendered: RenderedOutput):
        """Update session context with current interaction."""
        interaction = {
            "timestamp": asyncio.get_event_loop().time(),
            "input": raw_input,
            "interpreted": interpreted,
            "processing_result": processing_result,
            "output_mode": rendered.mode.value,
            "confidence": interpreted.get("confidence", 0.0)
        }
        
        # Maintain rolling context window
        if "interactions" not in self.session_context:
            self.session_context["interactions"] = []
        
        self.session_context["interactions"].append(interaction)
        
        # Keep only last 10 interactions to prevent context bloat
        if len(self.session_context["interactions"]) > 10:
            self.session_context["interactions"] = self.session_context["interactions"][-10:]
        
        # Update mode history
        self.mode_history.append(rendered.mode)
        if len(self.mode_history) > 20:
            self.mode_history = self.mode_history[-20:]
    
    def add_renderer(self, output_mode: OutputMode, renderer: OutputRenderer):
        """Add or replace a renderer for a specific output mode."""
        self.renderers[output_mode] = renderer
    
    def add_interpreter(self, input_mode: InputMode, interpreter: InputInterpreter):
        """Add or replace an interpreter for a specific input mode."""
        self.interpreters[input_mode] = interpreter
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session."""
        interactions = self.session_context.get("interactions", [])
        
        if not interactions:
            return {"total_interactions": 0}
        
        # Calculate mode distribution
        input_modes = {}
        output_modes = {}
        avg_confidence = 0
        
        for interaction in interactions:
            # Count output modes
            output_mode = interaction["output_mode"]
            output_modes[output_mode] = output_modes.get(output_mode, 0) + 1
            
            # Calculate average confidence
            avg_confidence += interaction["confidence"]
        
        avg_confidence /= len(interactions)
        
        return {
            "total_interactions": len(interactions),
            "output_mode_distribution": output_modes,
            "average_confidence": avg_confidence,
            "session_duration": interactions[-1]["timestamp"] - interactions[0]["timestamp"]
        }

# Example usage integration
async def demo_multimodal_router():
    """Demonstrate the multimodal context router."""
    # Initialize with existing components
    from microagents_conversational_ai.registry import agent_registry
    from microagents_conversational_ai.tokenizer.action_tokenizer import ActionTokenizer
    
    tokenizer = ActionTokenizer()
    router = MultimodalContextRouter(
        microagent_registry=agent_registry,
        tokenizer=tokenizer
    )
    
    # Process various types of input
    inputs = [
        "Show me all available agents",
        "$ ls -la /home/user",
        "Create a web scraper for reddit",
        "Generate a diagram of the microagent ecosystem",
        '{"query": "find security tools", "format": "table"}'
    ]
    
    for user_input in inputs:
        print(f"\n🔄 Processing: {user_input}")
        result = await router.process_input(user_input)
        print(f"📊 Output Mode: {result.mode.value}")
        print(f"📝 Content: {result.content}")
        if result.interactive_elements:
            print(f"🎛️  Interactive Elements: {len(result.interactive_elements)}")
        print(f"💡 Follow-ups: {', '.join(result.follow_up_suggestions)}")
    
    # Show session statistics
    stats = router.get_session_stats()
    print(f"\n📈 Session Stats: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    asyncio.run(demo_multimodal_router())
