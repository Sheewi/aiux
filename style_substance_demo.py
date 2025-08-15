"""
Style-Substance Separation Implementation Guide
Practical examples of capturing style while maintaining independent reasoning
"""

from style_aware_vision_mode import StyleAwareVisionMode
import asyncio

class StyleSubstanceDemo:
    """Demonstrates the dual-layer approach with real examples"""
    
    def __init__(self):
        self.examples = self._generate_examples()
    
    def _generate_examples(self):
        """Generate examples showing style vs substance separation"""
        
        return {
            "surface_mirror_only": {
                "description": "Pure style mimicry (what we DON'T want)",
                "user_input": "🎯 This dashboard is perfect! Everyone loves it. Just make it faster.",
                "bad_response": "🎯 Absolutely! This dashboard is indeed perfect! Everyone will love the faster version. Just making it faster is exactly what we need.",
                "problems": [
                    "Mirrors assumptions without questioning",
                    "Ignores potential issues with 'perfect' claim",
                    "Doesn't challenge vague 'make it faster' request",
                    "No independent analysis or alternatives"
                ]
            },
            
            "substance_independent": {
                "description": "Independent reasoning without style matching",
                "user_input": "🎯 This dashboard is perfect! Everyone loves it. Just make it faster.",
                "response": """
                Based on performance analysis, there are several optimization opportunities:
                
                1. Bundle size reduction through code splitting
                2. Database query optimization for data loading
                3. Caching strategies for frequently accessed data
                4. Lazy loading for non-critical components
                
                However, user feedback suggests interface confusion in key workflows. 
                Consider UX improvements alongside performance optimization.
                """,
                "strengths": [
                    "Independent technical analysis",
                    "Challenges assumptions about user satisfaction",
                    "Provides specific optimization strategies",
                    "Considers broader user experience"
                ],
                "weaknesses": [
                    "Ignores user's communication style",
                    "May feel impersonal or disconnected",
                    "Doesn't match user's goal-oriented tone"
                ]
            },
            
            "hybrid_aligned": {
                "description": "Style capture + independent reasoning (ideal)",
                "user_input": "🎯 This dashboard is perfect! Everyone loves it. Just make it faster.",
                "response": """
                🎯 **Performance Enhancement Strategy**
                
                Here's how we can boost dashboard speed while maintaining what's working:
                
                **Quick Wins:**
                1. Database query optimization → 40% faster data loading
                2. Component lazy loading → 60% faster initial render  
                3. Smart caching → Near-instant repeat visits
                
                **Reality Check:** 
                ⚠️ **User feedback analysis** shows some workflow confusion despite overall positive sentiment. Consider parallel UX improvements.
                
                **Alternative Approaches:**
                - Performance-first: Pure speed optimization
                - Holistic: Speed + UX improvements combined
                - Staged: Speed now, UX refinements next sprint
                
                **Bottom Line:** Fast performance + addressing hidden UX issues = truly optimal dashboard.
                """,
                "strengths": [
                    "Matches user's goal-oriented emoji style",
                    "Uses direct, action-focused language",
                    "Provides specific metrics (40%, 60%)",
                    "Challenges assumptions with evidence",
                    "Offers multiple approaches",
                    "Maintains user's 'bottom line' preference"
                ]
            }
        }
    
    async def demonstrate_dual_layer_processing(self):
        """Show the dual-layer approach in action"""
        
        print("🧠 Dual-Layer Processing Demonstration")
        print("=" * 60)
        
        # Initialize style-aware system
        system = StyleAwareVisionMode()
        
        # Example user request
        user_request = """
        🎯 The current analytics are basically useless. Everyone's complaining 
        they can't find actionable insights. We need to fix this ASAP.
        
        Here's what I'm thinking:
        1. Better chart types
        2. Clearer data summaries  
        3. Export functionality
        
        This should eliminate user frustration completely.
        """
        
        print("User Request:")
        print(user_request)
        print("\n" + "=" * 60)
        
        # Process with dual-layer approach
        result = await system.process_request_with_style_awareness(user_request)
        
        print("📊 Layer 1: Style Analysis")
        print("=" * 30)
        style_profile = result["style_profile_used"]
        print(f"Communication Style: {style_profile['communication_style']}")
        print(f"Technical Depth: {style_profile['avg_technical_depth']:.1f}")
        print(f"Directness Level: {style_profile['avg_directness']:.1f}")
        print(f"Preferred Structures: {', '.join(style_profile['preferred_structures'])}")
        
        print("\n🔬 Layer 2: Independent Reasoning")
        print("=" * 35)
        print("Autonomous Actions Executed:")
        for action in result["autonomous_actions"]:
            print(f"  • {action['intent_type']}")
            if 'critiques_applied' in action:
                print(f"    Critiques: {len(action['critiques_applied'])} reality checks applied")
            if 'alternatives_considered' in action:
                print(f"    Alternatives: {len(action['alternatives_considered'])} perspectives considered")
        
        print("\n🎯 Final Synthesized Response:")
        print("=" * 35)
        print(result["style_aware_response"])
        
        return result
    
    def show_style_vs_substance_examples(self):
        """Display examples of different approaches"""
        
        print("\n📚 Style vs Substance Examples")
        print("=" * 50)
        
        for approach, example in self.examples.items():
            print(f"\n## {approach.replace('_', ' ').title()}")
            print(f"**{example['description']}**")
            print(f"\n**User Input:** {example['user_input']}")
            
            if 'bad_response' in example:
                print(f"\n**Response:** {example['bad_response']}")
                print(f"\n❌ **Problems:**")
                for problem in example['problems']:
                    print(f"   - {problem}")
            else:
                print(f"\n**Response:** {example['response']}")
                
                if 'strengths' in example:
                    print(f"\n✅ **Strengths:**")
                    for strength in example['strengths']:
                        print(f"   + {strength}")
                
                if 'weaknesses' in example:
                    print(f"\n⚠️ **Considerations:**")
                    for weakness in example['weaknesses']:
                        print(f"   - {weakness}")
    
    async def interactive_style_learning(self):
        """Interactive demo of style learning process"""
        
        print("\n🎓 Interactive Style Learning Demo")
        print("=" * 45)
        
        # Sample different communication styles
        style_samples = {
            "Executive Brief": [
                "Bottom line: we need faster deployment cycles. Current process takes too long.",
                "TL;DR: User acquisition is down 15%. Need immediate optimization.",
                "Key issue: dashboard performance is hurting user retention."
            ],
            
            "Technical Deep": [
                "The current React component architecture is causing unnecessary re-renders. We should implement useMemo and useCallback for optimization.",
                "Database query performance shows N+1 issues in the user analytics endpoint. Suggest implementing eager loading with JOIN queries.",
                "Memory leaks detected in the chart rendering library. Consider switching to a more performant visualization framework."
            ],
            
            "Problem Solver": [
                "Here's the issue: users can't find what they need. Solution: redesign the navigation hierarchy.",
                "Problem identified: slow API responses. Root cause: inefficient database indexing. Fix: implement proper indexing strategy.",
                "Challenge: inconsistent data across dashboards. Approach: create single source of truth with real-time sync."
            ]
        }
        
        system = StyleAwareVisionMode()
        
        print("Learning from different communication styles...\n")
        
        for style_name, samples in style_samples.items():
            print(f"📊 Analyzing {style_name} Style:")
            
            # Learn style from samples
            profile = await system.bootstrap_style_learning(samples)
            
            print(f"   Style Classification: {profile['communication_style']}")
            print(f"   Technical Depth: {profile['avg_technical_depth']:.2f}")
            print(f"   Directness: {profile['avg_directness']:.2f}")
            print(f"   Key Patterns: {', '.join(profile['tone_patterns'][:3])}")
            print()

async def main():
    """Run the complete style-substance separation demo"""
    
    print("🚀 Style-Substance Separation Architecture Demo")
    print("=" * 65)
    
    demo = StyleSubstanceDemo()
    
    # Show examples
    demo.show_style_vs_substance_examples()
    
    # Interactive style learning
    await demo.interactive_style_learning()
    
    # Full dual-layer processing demo
    await demo.demonstrate_dual_layer_processing()
    
    print("\n" + "=" * 65)
    print("✅ Demo Complete: Style-Substance Separation Implemented")
    print("\n🎯 **Key Takeaways:**")
    print("   • Style capture: Maintains your voice and communication patterns")
    print("   • Substance independence: Ensures critical thinking and alternative perspectives") 
    print("   • Critique integration: Actively challenges assumptions and blind spots")
    print("   • Adaptive learning: Continuously improves style matching while enhancing reasoning")
    print("\n💡 **Result:** Your expressive style + enhanced situational awareness")

if __name__ == "__main__":
    asyncio.run(main())
