"""
Style-Substance Separation Architecture
Meta-alignment system that captures expressive style while maintaining independent reasoning
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

class ReasoningMode(Enum):
    """Types of reasoning approaches"""
    SURFACE_MIRROR = "surface_mirror"       # Copy style only
    SUBSTANCE_INDEPENDENT = "independent"   # Full independent reasoning
    HYBRID_ALIGNED = "hybrid"              # Style + enhanced reasoning
    CRITICAL_AUGMENT = "critical"          # Style + active critique

class CommunicationStyle(Enum):
    """Communication style patterns"""
    DIRECT = "direct"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"

@dataclass
class StyleProfile:
    """Captures user's communication style patterns"""
    tone_patterns: List[str] = field(default_factory=list)
    phrase_preferences: Dict[str, float] = field(default_factory=dict)
    formatting_style: Dict[str, Any] = field(default_factory=dict)
    humor_markers: List[str] = field(default_factory=list)
    technical_depth: float = 0.8
    directness_level: float = 0.7
    emoji_usage: bool = True
    structure_preferences: List[str] = field(default_factory=list)

@dataclass
class SubstanceCore:
    """Independent reasoning and knowledge base"""
    knowledge_sources: List[str] = field(default_factory=list)
    reasoning_chains: List[Dict] = field(default_factory=list)
    critique_points: List[str] = field(default_factory=list)
    alternative_perspectives: List[str] = field(default_factory=list)
    blind_spot_checks: List[str] = field(default_factory=list)

class MetaAlignmentEngine:
    """Engine that separates style capture from substance reasoning"""
    
    def __init__(self):
        self.style_profile = StyleProfile()
        self.substance_core = SubstanceCore()
        self.reasoning_mode = ReasoningMode.HYBRID_ALIGNED
        self.learning_history = []
        self.critique_enabled = True
        
    def analyze_user_style(self, user_input: str) -> StyleProfile:
        """Analyze and capture user's communication style"""
        
        # Tone analysis
        tone_patterns = []
        if "🎯" in user_input or "🚀" in user_input:
            tone_patterns.append("goal_oriented_visual")
        if "—" in user_input:
            tone_patterns.append("em_dash_emphasis")
        if re.search(r'\b(basically|essentially|fundamentally)\b', user_input.lower()):
            tone_patterns.append("foundational_framing")
        if re.search(r'\b(here\'s how|here\'s what)\b', user_input.lower()):
            tone_patterns.append("solution_oriented")
        
        # Phrase preferences
        phrase_scores = {}
        technical_phrases = ["implementation", "architecture", "systematic", "framework"]
        direct_phrases = ["cut to the chase", "bottom line", "key point"]
        creative_phrases = ["innovative", "game-changing", "paradigm shift"]
        
        for phrase in technical_phrases:
            if phrase.lower() in user_input.lower():
                phrase_scores[phrase] = phrase_scores.get(phrase, 0) + 0.8
                
        # Formatting analysis
        formatting_style = {
            "uses_bullets": "•" in user_input or "-" in user_input,
            "uses_numbers": bool(re.search(r'\\d+\\.', user_input)),
            "uses_headers": "#" in user_input,
            "uses_emphasis": "**" in user_input or "*" in user_input,
            "uses_code_blocks": "```" in user_input or "`" in user_input
        }
        
        # Structure preferences
        structure_prefs = []
        if re.search(r'\b(first|second|third|1\.|2\.|3\.)\b', user_input.lower()):
            structure_prefs.append("numbered_progression")
        if "TL;DR" in user_input:
            structure_prefs.append("summary_focused")
        if re.search(r'\b(example|for instance|like)\b', user_input.lower()):
            structure_prefs.append("example_heavy")
        
        return StyleProfile(
            tone_patterns=tone_patterns,
            phrase_preferences=phrase_scores,
            formatting_style=formatting_style,
            structure_preferences=structure_prefs,
            technical_depth=0.9 if any(p in user_input.lower() for p in technical_phrases) else 0.6,
            directness_level=0.8 if any(p in user_input.lower() for p in direct_phrases) else 0.6,
            emoji_usage="🎯" in user_input or "🚀" in user_input
        )
    
    def generate_independent_reasoning(self, topic: str, context: Dict[str, Any]) -> SubstanceCore:
        """Generate independent reasoning separate from user's perspective"""
        
        # Knowledge source diversification
        knowledge_sources = [
            "technical_documentation",
            "academic_research", 
            "industry_best_practices",
            "alternative_methodologies",
            "failure_case_studies",
            "cross_domain_solutions"
        ]
        
        # Independent reasoning chains
        reasoning_chains = [
            {
                "chain_type": "first_principles",
                "steps": [
                    "What are the fundamental constraints?",
                    "What assumptions might be false?",
                    "What would a completely different approach look like?"
                ]
            },
            {
                "chain_type": "red_team_analysis", 
                "steps": [
                    "What could go wrong with this approach?",
                    "What are the hidden costs?",
                    "Who might this negatively impact?"
                ]
            },
            {
                "chain_type": "opportunity_cost",
                "steps": [
                    "What are we NOT doing by choosing this path?",
                    "What resources could be better allocated?",
                    "What simpler solutions exist?"
                ]
            }
        ]
        
        # Critical perspective points
        critique_points = [
            "Is this solving the real problem or a symptom?",
            "Are we over-engineering a simple solution?", 
            "What biases might be influencing this approach?",
            "Have we considered accessibility and inclusion?",
            "What's the maintenance burden of this solution?",
            "Are there proven alternatives we're ignoring?"
        ]
        
        # Alternative perspectives
        alternative_perspectives = [
            "User experience perspective",
            "System maintenance perspective", 
            "Security/risk perspective",
            "Cost-efficiency perspective",
            "Scalability perspective",
            "Competitor perspective"
        ]
        
        # Blind spot identification
        blind_spot_checks = [
            "Technical debt implications",
            "Team capability requirements",
            "Timeline reality check",
            "Resource availability",
            "Market timing",
            "Regulatory considerations"
        ]
        
        return SubstanceCore(
            knowledge_sources=knowledge_sources,
            reasoning_chains=reasoning_chains,
            critique_points=critique_points,
            alternative_perspectives=alternative_perspectives,
            blind_spot_checks=blind_spot_checks
        )
    
    def synthesize_style_substance(self, user_request: str, enhanced_reasoning: SubstanceCore) -> str:
        """Combine user's style with independent reasoning"""
        
        # Analyze user's style from the request
        style = self.analyze_user_style(user_request)
        
        # Generate response structure based on style preferences
        response_parts = []
        
        # Opening - match user's directness and tone
        if style.directness_level > 0.7:
            if style.emoji_usage:
                response_parts.append("🎯 **Direct Analysis & Enhanced Approach**")
            else:
                response_parts.append("**Direct Analysis & Enhanced Approach**")
        else:
            response_parts.append("Here's a comprehensive analysis with enhanced reasoning:")
        
        # Main content - structured according to user preferences
        if "numbered_progression" in style.structure_preferences:
            response_parts.append("\n## 1. Style-Substance Dual Layer Implementation")
            response_parts.append("\n## 2. Independent Reasoning Enhancement")
            response_parts.append("\n## 3. Critical Perspective Integration")
            response_parts.append("\n## 4. Blind Spot Mitigation")
        else:
            response_parts.append("\n## Enhanced Analysis Framework")
        
        # Technical depth matching
        if style.technical_depth > 0.8:
            response_parts.append("\n### Architectural Implementation:")
            response_parts.append("```python")
            response_parts.append("# Dual-layer processing architecture")
            response_parts.append("class StyleSubstanceSeparator:")
            response_parts.append("    def process(self, input_text):")
            response_parts.append("        style_layer = self.extract_style(input_text)")
            response_parts.append("        substance_layer = self.reason_independently(input_text)")
            response_parts.append("        return self.synthesize(style_layer, substance_layer)")
            response_parts.append("```")
        
        # Critical reasoning integration
        response_parts.append("\n### Independent Reasoning Components:")
        for i, reasoning_chain in enumerate(enhanced_reasoning.reasoning_chains, 1):
            response_parts.append(f"\n**{i}. {reasoning_chain['chain_type'].replace('_', ' ').title()}:**")
            for step in reasoning_chain['steps']:
                response_parts.append(f"   - {step}")
        
        # Critique and alternatives
        response_parts.append("\n### Critical Perspective Points:")
        for critique in enhanced_reasoning.critique_points[:3]:  # Top 3 critiques
            response_parts.append(f"- **{critique}**")
        
        # Blind spot checks
        response_parts.append("\n### Blind Spot Mitigation:")
        for blind_spot in enhanced_reasoning.blind_spot_checks[:3]:
            response_parts.append(f"⚠️ **{blind_spot}** - Requires additional analysis")
        
        # Conclusion - match user's style
        if style.directness_level > 0.7:
            if "summary_focused" in style.structure_preferences:
                response_parts.append("\n**TL;DR**: Your style + independent reasoning + active critique = enhanced decision-making without blind spots.")
            else:
                response_parts.append("\n**Bottom Line**: This approach captures your communication style while ensuring the AI thinks independently and challenges assumptions.")
        
        return "\n".join(response_parts)
    
    def enable_critique_mode(self, user_input: str) -> Tuple[str, List[str]]:
        """Generate response with active critique and alternative suggestions"""
        
        critiques = []
        
        # Analyze potential issues
        if "just" in user_input.lower() or "simply" in user_input.lower():
            critiques.append("🚨 **Complexity Warning**: Words like 'just' or 'simply' often hide complexity. Real implementation may be more nuanced.")
        
        if "everyone" in user_input.lower() or "all users" in user_input.lower():
            critiques.append("🎯 **Universality Check**: 'Everyone' assumptions often miss edge cases. Consider accessibility, different skill levels, and varied use cases.")
        
        if "best practice" in user_input.lower():
            critiques.append("🔍 **Best Practice Reality**: What's 'best' depends on context. Consider trade-offs, constraints, and specific organizational needs.")
        
        if "should" in user_input.lower():
            critiques.append("⚖️ **Normative Language**: 'Should' statements often reflect preferences rather than requirements. Consider: what are the actual constraints?")
        
        # Generate enhanced response with critiques
        enhanced_reasoning = self.generate_independent_reasoning(user_input, {})
        base_response = self.synthesize_style_substance(user_input, enhanced_reasoning)
        
        # Add critique section
        if critiques:
            critique_section = "\n\n## 🔬 Critical Analysis & Reality Checks"
            for critique in critiques:
                critique_section += f"\n{critique}"
            
            critique_section += "\n\n## 🌟 Alternative Approaches to Consider:"
            for alt in enhanced_reasoning.alternative_perspectives[:3]:
                critique_section += f"\n- **{alt}**: How would this change our approach?"
        
            base_response += critique_section
        
        return base_response, critiques
    
    def learn_from_interaction(self, user_input: str, ai_response: str, user_feedback: Optional[str] = None):
        """Learn and adapt from user interactions"""
        
        interaction_log = {
            "timestamp": asyncio.get_event_loop().time(),
            "user_style_detected": self.analyze_user_style(user_input).__dict__,
            "reasoning_mode_used": self.reasoning_mode.value,
            "critiques_generated": self.critique_enabled,
            "user_feedback": user_feedback,
            "adaptation_needed": user_feedback and "not quite" in user_feedback.lower()
        }
        
        self.learning_history.append(interaction_log)
        
        # Adapt based on feedback
        if user_feedback:
            if "too technical" in user_feedback.lower():
                self.style_profile.technical_depth = max(0.3, self.style_profile.technical_depth - 0.2)
            elif "more detail" in user_feedback.lower():
                self.style_profile.technical_depth = min(1.0, self.style_profile.technical_depth + 0.2)
            
            if "too formal" in user_feedback.lower():
                self.style_profile.directness_level = min(1.0, self.style_profile.directness_level + 0.1)

# Implementation for your specific dashboard context
class DashboardMetaAlignment(MetaAlignmentEngine):
    """Specialized meta-alignment for dashboard development"""
    
    def __init__(self):
        super().__init__()
        self.domain_context = "customer_dashboard_development"
        
    def process_dashboard_request(self, user_request: str) -> str:
        """Process dashboard-related requests with style-substance separation"""
        
        # Generate enhanced reasoning specific to dashboard context
        enhanced_reasoning = self.generate_dashboard_reasoning(user_request)
        
        # Apply critique mode if enabled
        if self.critique_enabled:
            response, critiques = self.enable_critique_mode(user_request)
            return response
        else:
            return self.synthesize_style_substance(user_request, enhanced_reasoning)
    
    def generate_dashboard_reasoning(self, request: str) -> SubstanceCore:
        """Generate dashboard-specific independent reasoning"""
        
        # Dashboard-specific knowledge sources
        knowledge_sources = [
            "UX research on dashboard effectiveness",
            "Performance optimization case studies", 
            "Accessibility guidelines for data visualization",
            "Cross-browser compatibility requirements",
            "Mobile-first design principles",
            "Real-time data handling best practices"
        ]
        
        # Dashboard-specific reasoning chains
        reasoning_chains = [
            {
                "chain_type": "user_task_analysis",
                "steps": [
                    "What specific tasks do users need to complete?",
                    "What's the cognitive load of the current interface?",
                    "How can we reduce decision fatigue?"
                ]
            },
            {
                "chain_type": "data_flow_optimization",
                "steps": [
                    "What's the critical path for data updates?",
                    "Where are the performance bottlenecks?",
                    "How can we make stale data obvious to users?"
                ]
            },
            {
                "chain_type": "scalability_analysis",
                "steps": [
                    "How will this perform with 10x more data?",
                    "What happens when the team grows?",
                    "How will this evolve with changing requirements?"
                ]
            }
        ]
        
        # Dashboard-specific critiques
        critique_points = [
            "Are we showing data or enabling decisions?",
            "Is this dashboard optimized for executives or operators?",
            "Are we measuring vanity metrics or actionable insights?",
            "Does the visual hierarchy match task priority?",
            "Are we assuming users understand our mental model?"
        ]
        
        return SubstanceCore(
            knowledge_sources=knowledge_sources,
            reasoning_chains=reasoning_chains,
            critique_points=critique_points,
            alternative_perspectives=[
                "End-user daily workflow perspective",
                "Data analyst efficiency perspective",
                "Executive summary perspective", 
                "Mobile user perspective",
                "Accessibility user perspective"
            ],
            blind_spot_checks=[
                "Performance with large datasets",
                "Internationalization requirements",
                "Offline functionality needs",
                "Integration complexity",
                "Training and adoption curve"
            ]
        )

# Example usage
async def demo_meta_alignment():
    """Demonstrate the meta-alignment system"""
    
    print("🧠 Meta-Alignment Engine Demo")
    print("=" * 50)
    
    # Initialize the system
    engine = DashboardMetaAlignment()
    
    # Simulate user request with style patterns
    user_request = """
    🎯 Here's what I need: enhanced dashboard performance that basically 
    eliminates loading delays. Everyone should see instant updates.
    
    Implementation approach:
    1. Real-time data streaming  
    2. Optimized component rendering
    3. Smart caching strategy
    
    This should be a game-changing improvement to user experience.
    """
    
    print("User Request (Style Analysis):")
    print(user_request)
    
    # Process with meta-alignment
    response = engine.process_dashboard_request(user_request)
    
    print("\n" + "=" * 50)
    print("Meta-Aligned Response:")
    print("=" * 50)
    print(response)
    
    # Show learning
    engine.learn_from_interaction(user_request, response, "Great analysis, but maybe less technical detail")
    
    print("\n📚 Learning History:")
    print(f"Interactions logged: {len(engine.learning_history)}")
    print(f"Technical depth adjusted to: {engine.style_profile.technical_depth}")

if __name__ == "__main__":
    asyncio.run(demo_meta_alignment())
