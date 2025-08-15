"""
Grok-like Output Format Specifications
Defines the structured output format with metadata tags for dynamic rendering.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json

class RenderType(Enum):
    """Available render types matching Grok's capabilities."""
    TEXT = "text"
    CLI = "cli"
    GRAPH = "graph"
    DIAGRAM = "diagram"
    CODE = "code"
    INTERACTIVE = "interactive"
    VISUALIZATION = "visualization"
    TABLE = "table"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON_VIEWER = "json_viewer"
    TREE = "tree"
    TERMINAL = "terminal"

@dataclass
class GrokStyleMetadata:
    """Metadata structure mimicking Grok's internal format."""
    render_type: RenderType
    confidence: float
    processing_time_ms: float
    context_tokens_used: int
    agent_suggestions: List[str]
    interactive_capabilities: List[str]
    follow_up_context: Dict[str, Any]
    execution_environment: Dict[str, Any]

@dataclass
class InteractiveElement:
    """Interactive UI elements that can be embedded in responses."""
    element_type: str  # button, input, dropdown, slider, etc.
    element_id: str
    label: str
    action: str
    parameters: Dict[str, Any]
    styling: Optional[Dict[str, Any]] = None

@dataclass
class GrokStyleOutput:
    """Complete output structure matching Grok's format."""
    content: Any
    metadata: GrokStyleMetadata
    interactive_elements: List[InteractiveElement]
    follow_up_suggestions: List[str]
    context_preservation: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "metadata": {
                "render_type": self.metadata.render_type.value,
                "confidence": self.metadata.confidence,
                "processing_time_ms": self.metadata.processing_time_ms,
                "context_tokens_used": self.metadata.context_tokens_used,
                "agent_suggestions": self.metadata.agent_suggestions,
                "interactive_capabilities": self.metadata.interactive_capabilities,
                "follow_up_context": self.metadata.follow_up_context,
                "execution_environment": self.metadata.execution_environment
            },
            "interactive_elements": [asdict(elem) for elem in self.interactive_elements],
            "follow_up_suggestions": self.follow_up_suggestions,
            "context_preservation": self.context_preservation
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

# Example output templates for different modes
GROK_OUTPUT_TEMPLATES = {
    "text_response": {
        "template": {
            "content": "response_content",
            "metadata": {
                "render_type": "text",
                "confidence": 0.95,
                "processing_time_ms": 150,
                "context_tokens_used": 500,
                "agent_suggestions": [],
                "interactive_capabilities": ["copy", "share", "expand"],
                "follow_up_context": {"conversation_state": "active"},
                "execution_environment": {"mode": "chat", "user_context": {}}
            },
            "interactive_elements": [
                {
                    "element_type": "button",
                    "element_id": "copy_response",
                    "label": "Copy",
                    "action": "copy_to_clipboard",
                    "parameters": {"target": "response_text"}
                }
            ],
            "follow_up_suggestions": [
                "Ask for more details",
                "Request a different format",
                "Continue the conversation"
            ],
            "context_preservation": {
                "last_intent": "general_query",
                "user_preferences": {},
                "session_state": {}
            }
        }
    },
    
    "cli_response": {
        "template": {
            "content": {
                "command": "sample_command",
                "output": "command_output_result",
                "exit_code": 0,
                "execution_time": "100ms"
            },
            "metadata": {
                "render_type": "cli",
                "confidence": 0.88,
                "processing_time_ms": 200,
                "context_tokens_used": 300,
                "agent_suggestions": ["terminal_agent", "command_validator"],
                "interactive_capabilities": ["execute", "modify", "copy", "save"],
                "follow_up_context": {"terminal_session": "active"},
                "execution_environment": {
                    "mode": "cli",
                    "shell": "bash",
                    "working_directory": "/current/path"
                }
            },
            "interactive_elements": [
                {
                    "element_type": "button",
                    "element_id": "execute_command",
                    "label": "▶ Execute",
                    "action": "run_command",
                    "parameters": {"command": "sample_command", "safe_mode": True},
                    "styling": {"color": "green", "icon": "play"}
                },
                {
                    "element_type": "input",
                    "element_id": "modify_command",
                    "label": "Modify command",
                    "action": "update_command",
                    "parameters": {"current_value": "sample_command"}
                }
            ],
            "follow_up_suggestions": [
                "Execute this command",
                "Modify parameters",
                "Show command documentation",
                "Add to script"
            ],
            "context_preservation": {
                "last_intent": "command_execution",
                "terminal_context": {"shell": "bash", "env_vars": {}},
                "command_history": []
            }
        }
    },
    
    "graph_response": {
        "template": {
            "content": {
                "graph_type": "network",
                "nodes": [
                    {"id": "node1", "label": "Agent Registry", "type": "system"},
                    {"id": "node2", "label": "Tokenizer", "type": "processor"},
                    {"id": "node3", "label": "Router", "type": "controller"}
                ],
                "edges": [
                    {"source": "node1", "target": "node3", "type": "data_flow"},
                    {"source": "node2", "target": "node3", "type": "processing"}
                ],
                "layout": "force_directed",
                "styling": {"node_color": "#4CAF50", "edge_color": "#2196F3"}
            },
            "metadata": {
                "render_type": "graph",
                "confidence": 0.92,
                "processing_time_ms": 800,
                "context_tokens_used": 1200,
                "agent_suggestions": ["graph_analyzer", "network_mapper", "visualization_agent"],
                "interactive_capabilities": ["zoom", "pan", "filter", "export", "annotate"],
                "follow_up_context": {"visualization_mode": "interactive"},
                "execution_environment": {
                    "mode": "visualization",
                    "render_engine": "d3.js",
                    "data_source": "processed"
                }
            },
            "interactive_elements": [
                {
                    "element_type": "dropdown",
                    "element_id": "layout_selector",
                    "label": "Layout",
                    "action": "change_layout",
                    "parameters": {
                        "options": ["force_directed", "circular", "hierarchical", "grid"],
                        "current": "force_directed"
                    }
                },
                {
                    "element_type": "slider",
                    "element_id": "zoom_control",
                    "label": "Zoom",
                    "action": "adjust_zoom",
                    "parameters": {"min": 0.1, "max": 5.0, "current": 1.0, "step": 0.1}
                },
                {
                    "element_type": "button",
                    "element_id": "export_graph",
                    "label": "💾 Export",
                    "action": "export_visualization",
                    "parameters": {"formats": ["svg", "png", "pdf", "json"]}
                }
            ],
            "follow_up_suggestions": [
                "Analyze node relationships",
                "Filter by node type",
                "Export visualization",
                "Add annotations",
                "Generate insights"
            ],
            "context_preservation": {
                "last_intent": "visualization_request",
                "graph_context": {"data_type": "network", "filters": {}},
                "visualization_state": {"zoom": 1.0, "pan": {"x": 0, "y": 0}}
            }
        }
    },
    
    "diagram_response": {
        "template": {
            "content": {
                "diagram_type": "mermaid",
                "source": "graph TD\n    A[Input] --> B[Process]\n    B --> C[Output]",
                "rendered_svg": None,
                "editable": True
            },
            "metadata": {
                "render_type": "diagram",
                "confidence": 0.85,
                "processing_time_ms": 600,
                "context_tokens_used": 800,
                "agent_suggestions": ["diagram_generator", "flowchart_agent", "uml_agent"],
                "interactive_capabilities": ["edit", "export", "share", "validate"],
                "follow_up_context": {"diagram_editing": "enabled"},
                "execution_environment": {
                    "mode": "diagram_editor",
                    "syntax": "mermaid",
                    "validation": "enabled"
                }
            },
            "interactive_elements": [
                {
                    "element_type": "code_editor",
                    "element_id": "diagram_editor",
                    "label": "Edit Diagram",
                    "action": "update_diagram",
                    "parameters": {
                        "syntax": "mermaid",
                        "current_value": "graph TD\n    A[Input] --> B[Process]\n    B --> C[Output]",
                        "live_preview": True
                    }
                },
                {
                    "element_type": "dropdown",
                    "element_id": "diagram_type_selector",
                    "label": "Diagram Type",
                    "action": "change_diagram_type",
                    "parameters": {
                        "options": ["flowchart", "sequence", "class", "state", "gantt"],
                        "current": "flowchart"
                    }
                }
            ],
            "follow_up_suggestions": [
                "Edit diagram source",
                "Add more nodes",
                "Change diagram type",
                "Export as image",
                "Generate documentation"
            ],
            "context_preservation": {
                "last_intent": "diagram_generation",
                "diagram_context": {"type": "flowchart", "complexity": "medium"},
                "editor_state": {"cursor_position": 0, "selection": None}
            }
        }
    },
    
    "code_response": {
        "template": {
            "content": {
                "language": "python",
                "code": "def hello_world():\n    print('Hello, World!')\n    return True",
                "filename": "example.py",
                "executable": True
            },
            "metadata": {
                "render_type": "code",
                "confidence": 0.90,
                "processing_time_ms": 400,
                "context_tokens_used": 600,
                "agent_suggestions": ["code_generator", "syntax_validator", "code_executor"],
                "interactive_capabilities": ["execute", "edit", "debug", "format", "save"],
                "follow_up_context": {"code_editing": "enabled"},
                "execution_environment": {
                    "mode": "code_editor",
                    "language": "python",
                    "runtime_available": True
                }
            },
            "interactive_elements": [
                {
                    "element_type": "button",
                    "element_id": "run_code",
                    "label": "▶ Run",
                    "action": "execute_code",
                    "parameters": {"language": "python", "code": "def hello_world():\n    print('Hello, World!')\n    return True"},
                    "styling": {"color": "green", "icon": "play"}
                },
                {
                    "element_type": "button",
                    "element_id": "format_code",
                    "label": "🎨 Format",
                    "action": "format_code",
                    "parameters": {"language": "python"}
                },
                {
                    "element_type": "code_editor",
                    "element_id": "code_editor",
                    "label": "Edit Code",
                    "action": "update_code",
                    "parameters": {
                        "language": "python",
                        "current_value": "def hello_world():\n    print('Hello, World!')\n    return True",
                        "syntax_highlighting": True,
                        "auto_complete": True
                    }
                }
            ],
            "follow_up_suggestions": [
                "Execute the code",
                "Add error handling",
                "Optimize performance",
                "Add documentation",
                "Create tests"
            ],
            "context_preservation": {
                "last_intent": "code_generation",
                "code_context": {"language": "python", "purpose": "general"},
                "editor_state": {"modified": False, "saved": False}
            }
        }
    }
}

def create_grok_output(render_type: RenderType, content: Any, **kwargs) -> GrokStyleOutput:
    """
    Create a Grok-style output with proper metadata and interactive elements.
    """
    template_key = f"{render_type.value}_response"
    template = GROK_OUTPUT_TEMPLATES.get(template_key, GROK_OUTPUT_TEMPLATES["text_response"])
    
    # Create metadata
    metadata = GrokStyleMetadata(
        render_type=render_type,
        confidence=kwargs.get('confidence', 0.9),
        processing_time_ms=kwargs.get('processing_time_ms', 200),
        context_tokens_used=kwargs.get('context_tokens_used', 500),
        agent_suggestions=kwargs.get('agent_suggestions', []),
        interactive_capabilities=kwargs.get('interactive_capabilities', []),
        follow_up_context=kwargs.get('follow_up_context', {}),
        execution_environment=kwargs.get('execution_environment', {})
    )
    
    # Create interactive elements
    interactive_elements = []
    for elem_data in template["template"].get("interactive_elements", []):
        element = InteractiveElement(
            element_type=elem_data["element_type"],
            element_id=elem_data["element_id"],
            label=elem_data["label"],
            action=elem_data["action"],
            parameters=elem_data["parameters"],
            styling=elem_data.get("styling")
        )
        interactive_elements.append(element)
    
    return GrokStyleOutput(
        content=content,
        metadata=metadata,
        interactive_elements=interactive_elements,
        follow_up_suggestions=kwargs.get('follow_up_suggestions', template["template"]["follow_up_suggestions"]),
        context_preservation=kwargs.get('context_preservation', template["template"]["context_preservation"])
    )

# Example usage
if __name__ == "__main__":
    # Example 1: Text response
    text_output = create_grok_output(
        RenderType.TEXT,
        "Here are the available microagents in your system...",
        confidence=0.95,
        agent_suggestions=["registry_agent", "discovery_agent"]
    )
    print("Text Output Example:")
    print(text_output.to_json())
    print("\n" + "="*50 + "\n")
    
    # Example 2: CLI response
    cli_output = create_grok_output(
        RenderType.CLI,
        {
            "command": "python scraper.py --url https://example.com",
            "output": "Successfully scraped 150 items",
            "exit_code": 0,
            "execution_time": "2.3s"
        },
        confidence=0.88,
        agent_suggestions=["web_scraper", "data_processor"]
    )
    print("CLI Output Example:")
    print(cli_output.to_json())
