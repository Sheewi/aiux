#!/usr/bin/env python3
"""
Grok-like Multimodal AI Demo
Demonstrates dynamic output switching and input interpretation within chat sessions.

Example Output Metadata Format (like Grok):
{
    "type": "multimodal_response",
    "content": "...",
    "render_mode": "graph",
    "interactive_elements": [...],
    "follow_up_actions": [...],
    "metadata": {
        "confidence": 0.95,
        "agent_suggestions": [...],
        "execution_context": {...}
    }
}
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from multimodal_context_router import (
    MultimodalContextRouter, 
    OutputMode, 
    InputMode,
    RenderedOutput
)
from registry import agent_registry
from tokenizer.action_tokenizer import ActionTokenizer
from tokenizer.microagent_registry import MicroAgentRegistry

class GrokLikeDemo:
    """
    Demo class showcasing Grok-like functionality:
    - Hot-swappable output renderers
    - Dynamic input interpreters
    - Context-aware agent suggestions
    - Interactive elements in responses
    """
    
    def __init__(self):
        self.tokenizer = ActionTokenizer()
        self.microagent_registry = MicroAgentRegistry()
        self.router = MultimodalContextRouter(
            microagent_registry=agent_registry,
            tokenizer=self.tokenizer
        )
        
    async def run_interactive_demo(self):
        """Run an interactive demo session."""
        print("🚀 Grok-like Multimodal AI Demo")
        print("=" * 50)
        print("Features:")
        print("• Dynamic output switching (text → cli → graph → diagram)")
        print("• Context-aware input interpretation")
        print("• Hot-swappable renderers without leaving chat")
        print("• Agent suggestions based on intent")
        print("• Interactive elements in responses")
        print("\nType 'exit' to quit, 'stats' for session statistics")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'stats':
                    await self._show_session_stats()
                    continue
                elif not user_input:
                    continue
                
                # Process with multimodal router
                print("🔄 Processing...")
                result = await self.router.process_input(user_input)
                
                # Display result in Grok-like format
                await self._display_grok_response(user_input, result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def _display_grok_response(self, input_text: str, result: RenderedOutput):
        """Display response in Grok-like format with metadata."""
        print("\n" + "=" * 60)
        print(f"🎯 Output Mode: {result.mode.value.upper()}")
        print(f"📝 Content Type: {type(result.content).__name__}")
        print("=" * 60)
        
        # Main content
        print(f"\n📄 Response:")
        if result.mode == OutputMode.CLI:
            print(f"```bash\n{result.content}\n```")
        elif result.mode == OutputMode.GRAPH:
            print("📊 Graph Data:")
            print(json.dumps(result.content, indent=2))
        elif result.mode == OutputMode.DIAGRAM:
            print("📈 Diagram:")
            print(f"```mermaid\n{result.content}\n```")
        else:
            print(result.content)
        
        # Interactive elements (Grok-like buttons/actions)
        if result.interactive_elements:
            print("\n🎛️  Interactive Elements:")
            for i, element in enumerate(result.interactive_elements):
                element_type = element.get("type", "button")
                label = element.get("label", f"Action {i+1}")
                print(f"  [{i+1}] {element_type}: {label}")
        
        # Follow-up suggestions
        if result.follow_up_suggestions:
            print("\n💡 Follow-up Suggestions:")
            for i, suggestion in enumerate(result.follow_up_suggestions):
                print(f"  {i+1}. {suggestion}")
        
        # Metadata (like Grok's context info)
        if result.metadata:
            print("\n🔍 Context Metadata:")
            metadata_display = {
                k: v for k, v in result.metadata.items() 
                if k not in ['raw_content', 'processing_steps']
            }
            print(json.dumps(metadata_display, indent=2))
        
        # Agent suggestions if available
        await self._show_agent_suggestions(input_text)
    
    async def _show_agent_suggestions(self, input_text: str):
        """Show suggested agents based on input context."""
        # Simulate context extraction (in real implementation, this would use the router's interpreter)
        context = {
            'intent': 'general',
            'entities': [],
            'input_mode': 'natural_language',
            'suggested_output_mode': 'text'
        }
        
        # Simple intent detection for demo
        if any(word in input_text.lower() for word in ['scrape', 'crawl', 'web']):
            context['intent'] = 'scrape'
            context['entities'] = [{'type': 'web', 'value': 'website'}]
        elif any(word in input_text.lower() for word in ['analyze', 'examine', 'investigate']):
            context['intent'] = 'analyze'
        elif any(word in input_text.lower() for word in ['create', 'generate', 'build']):
            context['intent'] = 'create'
        
        suggestions = agent_registry.suggest_agents_for_context(context)
        
        if suggestions:
            print("\n🤖 Suggested Agents:")
            for i, suggestion in enumerate(suggestions[:3]):  # Show top 3
                name = suggestion['agent_name']
                score = suggestion['relevance_score']
                reason = suggestion['reason']
                print(f"  {i+1}. {name} (score: {score:.2f}) - {reason}")
    
    async def _show_session_stats(self):
        """Display session statistics."""
        stats = self.router.get_session_stats()
        print("\n📊 Session Statistics:")
        print("=" * 40)
        print(json.dumps(stats, indent=2))
    
    async def run_predefined_examples(self):
        """Run predefined examples to showcase different modes."""
        examples = [
            {
                "input": "Show me all web scraping agents",
                "description": "Agent discovery with table output"
            },
            {
                "input": "$ ls -la /home/user/projects",
                "description": "Command-line interface rendering"
            },
            {
                "input": "Create a network diagram of microagent connections",
                "description": "Diagram generation with mermaid"
            },
            {
                "input": "Analyze the performance metrics",
                "description": "Data analysis with visualization"
            },
            {
                "input": '{"query": "security tools", "format": "graph", "nodes": ["scanner", "analyzer"]}',
                "description": "Structured query with graph output"
            }
        ]
        
        print("🎬 Running Predefined Examples")
        print("=" * 50)
        
        for i, example in enumerate(examples):
            print(f"\n📝 Example {i+1}: {example['description']}")
            print(f"Input: {example['input']}")
            print("-" * 30)
            
            result = await self.router.process_input(example['input'])
            await self._display_grok_response(example['input'], result)
            
            if i < len(examples) - 1:
                input("\n⏸️  Press Enter to continue...")
    
    def demonstrate_hot_swapping(self):
        """Demonstrate hot-swappable renderers (Grok's key feature)."""
        print("\n🔥 Hot-Swappable Renderer Demo")
        print("=" * 40)
        
        # Create custom renderer
        from multimodal_context_router import OutputRenderer, RenderedOutput, OutputMode
        
        class CustomTableRenderer(OutputRenderer):
            def render(self, content, metadata):
                # Convert content to ASCII table
                if isinstance(content, str) and 'agents' in content.lower():
                    table_content = """
┌─────────────────┬──────────────┬─────────────┐
│ Agent Name      │ Type         │ Status      │
├─────────────────┼──────────────┼─────────────┤
│ Web Scraper     │ Automation   │ Active      │
│ Data Analyzer   │ Analysis     │ Active      │
│ Security Scanner│ Security     │ Ready       │
└─────────────────┴──────────────┴─────────────┘
"""
                else:
                    table_content = f"TABLE: {content}"
                
                return RenderedOutput(
                    content=table_content,
                    mode=OutputMode.TABLE,
                    metadata=metadata,
                    interactive_elements=[
                        {"type": "sort", "label": "Sort by Name"},
                        {"type": "filter", "label": "Filter by Type"}
                    ],
                    follow_up_suggestions=["Export to CSV", "Refresh data", "Show details"]
                )
            
            def can_handle(self, content_type, metadata):
                return True
        
        # Hot-swap the renderer
        print("🔧 Installing custom table renderer...")
        self.router.add_renderer(OutputMode.TABLE, CustomTableRenderer())
        print("✅ Custom renderer installed! Now try: 'Show me all agents'")

async def main():
    """Main demo function."""
    demo = GrokLikeDemo()
    
    print("Choose demo mode:")
    print("1. Interactive chat session")
    print("2. Predefined examples")
    print("3. Hot-swapping demo")
    print("4. All features")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        await demo.run_interactive_demo()
    elif choice == "2":
        await demo.run_predefined_examples()
    elif choice == "3":
        demo.demonstrate_hot_swapping()
        await demo.run_interactive_demo()
    elif choice == "4":
        print("\n🎯 Running complete feature demo...")
        demo.demonstrate_hot_swapping()
        print("\n" + "="*60)
        await demo.run_predefined_examples()
        print("\n" + "="*60)
        await demo.run_interactive_demo()
    else:
        print("Invalid choice. Running interactive demo...")
        await demo.run_interactive_demo()

if __name__ == "__main__":
    asyncio.run(main())
