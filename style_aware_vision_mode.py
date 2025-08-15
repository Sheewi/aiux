"""
Meta-Alignment Integration for Dashboard System
Connects style-substance separation to existing Vision Mode AI
"""

import asyncio
import json
from typing import Dict, Any, List
from meta_alignment_engine import DashboardMetaAlignment, ReasoningMode
from vision_mode_controller import DashboardVisionController

class StyleAwareVisionMode:
    """
    Enhanced Vision Mode that captures your style while maintaining independent reasoning
    """
    
    def __init__(self):
        self.meta_alignment = DashboardMetaAlignment()
        self.vision_controller = DashboardVisionController()
        self.style_learned = False
        self.user_style_profile = None
        
    async def bootstrap_style_learning(self, user_samples: List[str]):
        """Learn user's style from sample communications"""
        
        print("🧠 Learning your communication style...")
        
        # Analyze multiple samples to build robust style profile
        combined_analysis = {
            "tone_patterns": [],
            "phrase_preferences": {},
            "formatting_style": {},
            "technical_depth": [],
            "directness_level": [],
            "structure_preferences": []
        }
        
        for sample in user_samples:
            style = self.meta_alignment.analyze_user_style(sample)
            
            combined_analysis["tone_patterns"].extend(style.tone_patterns)
            combined_analysis["phrase_preferences"].update(style.phrase_preferences)
            combined_analysis["technical_depth"].append(style.technical_depth)
            combined_analysis["directness_level"].append(style.directness_level)
            combined_analysis["structure_preferences"].extend(style.structure_preferences)
        
        # Aggregate the analysis
        self.user_style_profile = {
            "tone_patterns": list(set(combined_analysis["tone_patterns"])),
            "avg_technical_depth": sum(combined_analysis["technical_depth"]) / len(combined_analysis["technical_depth"]),
            "avg_directness": sum(combined_analysis["directness_level"]) / len(combined_analysis["directness_level"]),
            "preferred_structures": list(set(combined_analysis["structure_preferences"])),
            "communication_style": self._classify_communication_style(combined_analysis)
        }
        
        self.style_learned = True
        
        print(f"✅ Style Profile Learned:")
        print(f"   🎯 Communication Style: {self.user_style_profile['communication_style']}")
        print(f"   🔧 Technical Depth: {self.user_style_profile['avg_technical_depth']:.1f}/1.0")
        print(f"   📏 Directness Level: {self.user_style_profile['avg_directness']:.1f}/1.0")
        print(f"   📋 Preferred Structures: {', '.join(self.user_style_profile['preferred_structures'])}")
        
        return self.user_style_profile
    
    def _classify_communication_style(self, analysis: Dict) -> str:
        """Classify overall communication style"""
        
        avg_tech = sum(analysis["technical_depth"]) / len(analysis["technical_depth"])
        avg_direct = sum(analysis["directness_level"]) / len(analysis["directness_level"])
        
        if avg_tech > 0.8 and avg_direct > 0.7:
            return "Technical-Direct"
        elif avg_tech > 0.7 and "example_heavy" in analysis["structure_preferences"]:
            return "Technical-Explanatory"
        elif avg_direct > 0.8 and "summary_focused" in analysis["structure_preferences"]:
            return "Executive-Brief"
        elif "solution_oriented" in analysis["tone_patterns"]:
            return "Problem-Solver"
        else:
            return "Balanced-Analytical"
    
    async def process_request_with_style_awareness(self, user_request: str) -> Dict[str, Any]:
        """Process user request with style-substance separation"""
        
        if not self.style_learned:
            # Learn style from this request
            await self.bootstrap_style_learning([user_request])
        
        # Generate meta-aligned response
        aligned_response = self.meta_alignment.process_dashboard_request(user_request)
        
        # Execute vision mode actions with style awareness
        vision_actions = await self.vision_controller.process_intent_with_style(
            user_request, 
            self.user_style_profile
        )
        
        # Combine style-aware response with autonomous actions
        result = {
            "style_aware_response": aligned_response,
            "autonomous_actions": vision_actions,
            "style_profile_used": self.user_style_profile,
            "reasoning_mode": self.meta_alignment.reasoning_mode.value,
            "critiques_enabled": self.meta_alignment.critique_enabled
        }
        
        return result
    
    async def enable_continuous_style_adaptation(self, feedback_enabled: bool = True):
        """Enable continuous learning from user interactions"""
        
        self.meta_alignment.critique_enabled = True
        self.feedback_enabled = feedback_enabled
        
        print("🔄 Continuous style adaptation enabled")
        print("   - Real-time critique mode: ON")
        print("   - Style adaptation learning: ON") 
        print("   - Independent reasoning: ENHANCED")
        
        return {
            "adaptation_mode": "continuous",
            "critique_mode": True,
            "learning_enabled": feedback_enabled
        }

# Enhanced Vision Controller with Style Awareness
class StyleAwareDashboardController(DashboardVisionController):
    """Vision controller enhanced with style-substance separation"""
    
    def __init__(self):
        super().__init__()
        self.style_profile = None
        
    async def process_intent_with_style(self, user_request: str, style_profile: Dict) -> List[Dict]:
        """Process intent while maintaining user's communication style in responses"""
        
        self.style_profile = style_profile
        
        # Generate intents with style awareness
        style_aware_intents = await self._generate_style_aware_intents(user_request)
        
        # Execute with enhanced reasoning
        results = []
        for intent in style_aware_intents:
            
            # Apply independent reasoning to each intent
            enhanced_intent = await self._apply_independent_reasoning(intent)
            
            # Execute the enhanced intent
            result = await self._execute_enhanced_intent(enhanced_intent)
            results.append(result)
        
        return results
    
    async def _generate_style_aware_intents(self, request: str) -> List[Dict]:
        """Generate intents that respect user's style while adding substance"""
        
        base_intents = [
            {
                "type": "dashboard_enhancement",
                "description": "Improve dashboard based on user request",
                "user_request": request,
                "style_considerations": self.style_profile
            }
        ]
        
        # Add style-specific intents
        if self.style_profile["communication_style"] == "Technical-Direct":
            base_intents.extend([
                {
                    "type": "technical_deep_dive",
                    "description": "Provide detailed technical implementation",
                    "focus": "architecture_and_performance"
                },
                {
                    "type": "direct_action_plan", 
                    "description": "Generate clear, actionable steps",
                    "focus": "implementation_roadmap"
                }
            ])
        
        elif self.style_profile["communication_style"] == "Problem-Solver":
            base_intents.extend([
                {
                    "type": "root_cause_analysis",
                    "description": "Identify underlying issues",
                    "focus": "systematic_problem_identification"
                },
                {
                    "type": "solution_alternatives",
                    "description": "Generate multiple solution approaches", 
                    "focus": "option_comparison"
                }
            ])
        
        return base_intents
    
    async def _apply_independent_reasoning(self, intent: Dict) -> Dict:
        """Apply independent reasoning to enhance intent"""
        
        enhanced_intent = intent.copy()
        
        # Add independent perspective checks
        enhanced_intent["independent_checks"] = [
            "What assumptions might be wrong?",
            "What are we not considering?", 
            "How could this fail?",
            "What would a different domain expert do?",
            "What's the simplest alternative?"
        ]
        
        # Add critique points
        enhanced_intent["critique_points"] = [
            "Is this solving the real problem?",
            "Are we over-engineering?",
            "What's the maintenance cost?",
            "How does this scale?",
            "What about edge cases?"
        ]
        
        # Add alternative perspectives
        enhanced_intent["alternative_perspectives"] = [
            "User experience impact",
            "Development team burden",
            "System performance implications",
            "Future scalability concerns",
            "Cost-benefit analysis"
        ]
        
        return enhanced_intent
    
    async def _execute_enhanced_intent(self, intent: Dict) -> Dict:
        """Execute intent with enhanced reasoning applied"""
        
        execution_result = {
            "intent_type": intent["type"],
            "style_aware_execution": True,
            "independent_reasoning_applied": True,
            "execution_timestamp": asyncio.get_event_loop().time()
        }
        
        # Simulate style-aware execution
        if intent["type"] == "technical_deep_dive":
            execution_result["actions"] = [
                "Generated detailed architecture analysis",
                "Provided performance optimization recommendations", 
                "Created implementation timeline with technical milestones",
                "Added code quality and testing considerations"
            ]
            
        elif intent["type"] == "direct_action_plan":
            execution_result["actions"] = [
                "Created prioritized action items",
                "Defined clear success criteria",
                "Estimated resource requirements",
                "Set realistic timelines with buffer"
            ]
            
        elif intent["type"] == "root_cause_analysis":
            execution_result["actions"] = [
                "Mapped problem symptoms to potential causes",
                "Identified systemic vs. surface-level issues",
                "Analyzed contributing factors",
                "Prioritized causes by impact and feasibility"
            ]
        
        # Apply critique to execution
        execution_result["critiques_applied"] = intent.get("critique_points", [])
        execution_result["alternatives_considered"] = intent.get("alternative_perspectives", [])
        
        return execution_result

# Demo implementation
async def demo_style_aware_system():
    """Demonstrate the style-aware meta-alignment system"""
    
    print("🚀 Style-Aware Meta-Alignment System Demo")
    print("=" * 60)
    
    # Initialize system
    style_aware_system = StyleAwareVisionMode()
    
    # Sample user communications for style learning
    user_samples = [
        """
        🎯 Here's what I need: enhanced dashboard performance that basically 
        eliminates loading delays. Everyone should see instant updates.
        
        Implementation approach:
        1. Real-time data streaming  
        2. Optimized component rendering
        3. Smart caching strategy
        """,
        """
        Bottom line: the current UI is confusing users. We need to simplify 
        the workflow and make critical actions more obvious. 
        
        TL;DR: Better UX = better user adoption.
        """,
        """
        Let's solve the scalability problem systematically:
        - Identify bottlenecks
        - Implement targeted optimizations  
        - Validate performance improvements
        
        This should be a game-changing improvement.
        """
    ]
    
    # Learn user style
    style_profile = await style_aware_system.bootstrap_style_learning(user_samples)
    
    # Enable continuous adaptation
    adaptation_config = await style_aware_system.enable_continuous_style_adaptation()
    
    # Process a new request with style awareness
    new_request = """
    The dashboard needs better data visualization. Users are missing 
    key insights because the charts aren't intuitive enough.
    """
    
    print("\n" + "=" * 60)
    print("Processing New Request with Style Awareness:")
    print("=" * 60)
    
    result = await style_aware_system.process_request_with_style_awareness(new_request)
    
    print("\n📊 Style-Aware Response:")
    print(result["style_aware_response"])
    
    print("\n🤖 Autonomous Actions Executed:")
    for action in result["autonomous_actions"]:
        print(f"   - {action['intent_type']}: {len(action.get('actions', []))} actions completed")
    
    print(f"\n🎨 Style Profile Applied: {result['style_profile_used']['communication_style']}")
    print(f"🧠 Reasoning Mode: {result['reasoning_mode']}")
    print(f"🔍 Critiques Enabled: {result['critiques_enabled']}")
    
    return result

if __name__ == "__main__":
    asyncio.run(demo_style_aware_system())
