#!/usr/bin/env python3
"""
Complete Grok-like System Demo
Shows all the features working together just like Grok's interface.
"""

import asyncio
import json
from multimodal_context_router import MultimodalContextRouter, OutputMode
from registry import agent_registry
from tokenizer.action_tokenizer import ActionTokenizer
from grok_output_format import create_grok_output, RenderType

class GrokLikeSystem:
    """Complete Grok-like system demonstration."""
    
    def __init__(self):
        self.router = MultimodalContextRouter(
            microagent_registry=agent_registry,
            tokenizer=ActionTokenizer()
        )
    
    async def demonstrate_mode_switching(self):
        """Demonstrate dynamic mode switching like Grok."""
        
        print("🎭 Grok-like Dynamic Mode Switching Demo")
        print("=" * 60)
        print("Watch how the system dynamically switches output modes")
        print("based on input content, just like Grok!")
        print("=" * 60)
        
        # Examples that trigger different modes
        examples = [
            {
                "input": "Show me available microagents",
                "expected": "TEXT → TABLE",
                "description": "Natural language → Agent discovery"
            },
            {
                "input": "$ ls -la | grep .py",
                "expected": "CLI",
                "description": "Command detection → Terminal interface"
            },
            {
                "input": "Create a flowchart of the agent architecture",
                "expected": "DIAGRAM",
                "description": "Visualization request → Diagram mode"
            },
            {
                "input": "Generate network graph of agent relationships",
                "expected": "GRAPH", 
                "description": "Graph request → Interactive visualization"
            },
            {
                "input": '{"action": "find", "type": "security_agents", "format": "table"}',
                "expected": "TABLE",
                "description": "Structured input → Formatted output"
            },
            {
                "input": "Write a Python function to scrape websites",
                "expected": "CODE",
                "description": "Code request → Interactive editor"
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\n🔄 Example {i}: {example['description']}")
            print(f"📝 Input: {example['input']}")
            print(f"🎯 Expected Mode: {example['expected']}")
            print("-" * 50)
            
            # Process input
            result = await self.router.process_input(example['input'])
            
            # Show Grok-like output
            print(f"✨ RENDERED AS: {result.mode.value.upper()}")
            
            # Show content preview
            if isinstance(result.content, dict):
                print(f"📊 Content: {json.dumps(result.content, indent=2)[:200]}...")
            else:
                print(f"📄 Content: {str(result.content)[:150]}...")
            
            # Show interactive elements (key Grok feature)
            if result.interactive_elements:
                print(f"🎛️  Interactive Elements:")
                for elem in result.interactive_elements[:3]:
                    print(f"   • {elem.element_type}: {elem.label}")
            
            # Show follow-up suggestions
            if result.follow_up_suggestions:
                print(f"💡 Follow-ups: {', '.join(result.follow_up_suggestions[:2])}")
            
            print("✅ Mode switch complete!")
            
            if i < len(examples):
                input("   ⏸️  Press Enter for next example...")
    
    def demonstrate_grok_output_format(self):
        """Show the structured output format that mimics Grok's internal format."""
        
        print("\n🏗️  Grok-style Output Format")
        print("=" * 50)
        print("This is the structured format with metadata tags")
        print("that enables dynamic rendering (like Grok's internals)")
        print("=" * 50)
        
        # Example CLI output in Grok format
        cli_output = create_grok_output(
            RenderType.CLI,
            {
                "command": "python -m microagents.web_scraper --url https://news.ycombinator.com",
                "output": "🕷️  Starting web scraper...\n✅ Scraped 30 articles\n💾 Saved to articles.json",
                "exit_code": 0,
                "execution_time": "2.4s"
            },
            confidence=0.95,
            processing_time_ms=340,
            agent_suggestions=["web_scraper", "data_processor", "file_manager"],
            interactive_capabilities=["execute", "modify", "schedule", "save"]
        )
        
        print("📋 Grok-style CLI Output:")
        print(cli_output.to_json())
        
        print("\n🎯 Key Features (just like Grok):")
        print("• 📊 Rich metadata (confidence, timing, context)")
        print("• 🎛️  Interactive elements (buttons, inputs, controls)")
        print("• 🤖 Agent suggestions (relevant microagents)")
        print("• 💡 Follow-up actions (contextual next steps)")
        print("• 🔄 Context preservation (session state)")
    
    async def demonstrate_hot_swapping(self):
        """Demonstrate hot-swappable renderers (Grok's key feature)."""
        
        print("\n🔥 Hot-swappable Renderers (Grok's Secret Sauce)")
        print("=" * 60)
        print("Add new output modes WITHOUT restarting the session!")
        print("=" * 60)
        
        # Create a custom renderer
        from multimodal_context_router import OutputRenderer, RenderedOutput
        
        class CustomASCIIRenderer(OutputRenderer):
            def render(self, content, metadata):
                # Convert content to ASCII art table
                ascii_content = f"""
╔══════════════════════════════════════╗
║           CUSTOM RENDERER            ║
╠══════════════════════════════════════╣
║ Content: {str(content)[:25]:<25} ║
║ Mode:    ASCII Art                   ║
║ Status:  🔥 HOT-SWAPPED!             ║
╚══════════════════════════════════════╝
"""
                return RenderedOutput(
                    content=ascii_content,
                    mode=OutputMode.TABLE,  # Use existing mode
                    metadata=metadata,
                    interactive_elements=[
                        {
                            "element_type": "button",
                            "element_id": "ascii_export",
                            "label": "📋 Copy ASCII",
                            "action": "copy_ascii",
                            "parameters": {"content": ascii_content}
                        }
                    ],
                    follow_up_suggestions=["Try another style", "Export as text", "Share ASCII art"]
                )
            
            def can_handle(self, content_type, metadata):
                return True
        
        # Hot-swap the renderer
        print("🔧 Installing custom ASCII renderer...")
        self.router.add_renderer(OutputMode.TABLE, CustomASCIIRenderer())
        print("✅ Custom renderer installed! (No restart needed)")
        
        # Test the new renderer
        print("\n🧪 Testing hot-swapped renderer...")
        result = await self.router.process_input('{"query": "agents", "format": "table"}')
        
        print(f"🎨 New output:")
        print(result.content)
        print("🔥 Hot-swap successful! Just like Grok!")
    
    async def show_session_context(self):
        """Show how context is preserved across mode switches."""
        
        print("\n🧠 Session Context Preservation")
        print("=" * 50)
        
        # Process several inputs to build context
        inputs = [
            "Find web scraping tools",
            "$ python scraper.py",
            "Show me the results as a graph"
        ]
        
        for inp in inputs:
            await self.router.process_input(inp)
        
        # Show preserved context
        stats = self.router.get_session_stats()
        print("📊 Session Statistics:")
        print(json.dumps(stats, indent=2))
        
        print("\n🎯 Context Features:")
        print("• 📈 Mode switching history")
        print("• 🎯 Confidence tracking")
        print("• ⏱️  Timing analysis")
        print("• 🔄 Interaction patterns")

async def main():
    """Run the complete Grok-like system demo."""
    
    print("🚀 Welcome to the Grok-like Multimodal AI System!")
    print("=" * 60)
    print("This system replicates Grok's core functionality:")
    print("• Dynamic output mode switching")
    print("• Hot-swappable renderers") 
    print("• Context-aware processing")
    print("• Interactive elements")
    print("• Session state preservation")
    print("=" * 60)
    
    system = GrokLikeSystem()
    
    print("\nSelect demo:")
    print("1. 🎭 Dynamic Mode Switching")
    print("2. 🏗️  Grok Output Format")
    print("3. 🔥 Hot-swappable Renderers")
    print("4. 🧠 Session Context")
    print("5. 🎪 Complete Demo (All features)")
    
    choice = input("\nEnter choice (1-5): ").strip() or "5"
    
    if choice == "1":
        await system.demonstrate_mode_switching()
    elif choice == "2":
        system.demonstrate_grok_output_format()
    elif choice == "3":
        await system.demonstrate_hot_swapping()
    elif choice == "4":
        await system.show_session_context()
    elif choice == "5":
        print("\n🎪 Running Complete Demo...")
        system.demonstrate_grok_output_format()
        input("\n⏸️  Press Enter to continue to mode switching...")
        await system.demonstrate_mode_switching()
        input("\n⏸️  Press Enter to continue to hot-swapping...")
        await system.demonstrate_hot_swapping()
        input("\n⏸️  Press Enter to see session context...")
        await system.show_session_context()
    else:
        print("Invalid choice, running complete demo...")
        await system.demonstrate_mode_switching()
    
    print("\n🎉 Demo Complete!")
    print("You now have a Grok-like system with:")
    print("✅ Dynamic mode switching")
    print("✅ Hot-swappable components") 
    print("✅ Rich metadata and context")
    print("✅ Interactive elements")
    print("✅ Agent orchestration")
    print("\nAll without Grok's sandbox - full user control! 🔓")

if __name__ == "__main__":
    asyncio.run(main())
