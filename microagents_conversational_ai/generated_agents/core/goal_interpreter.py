"""
Goal Interpreter - Conversational Interface Layer
Extracts objectives, constraints, and success criteria from natural language conversations.
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from core.base_agent import MicroAgent
from pydantic import BaseModel

@dataclass
class ExtractedGoal:
    """Structured representation of an extracted goal."""
    primary_objective: str
    key_constraints: List[str]
    success_criteria: List[str]
    preferred_solutions: List[str]
    urgency_level: int  # 1-10 scale
    complexity_score: float
    required_capabilities: List[str]
    estimated_duration: Optional[str]

class GoalInterpreterInput(BaseModel):
    conversation_history: str
    context: Optional[Dict[str, Any]] = {}
    user_preferences: Optional[Dict[str, Any]] = {}

class GoalInterpreterOutput(BaseModel):
    extracted_goal: Dict[str, Any]
    confidence_score: float
    clarification_questions: List[str]
    recommended_agents: List[str]
    execution_strategy: str

class GoalInterpreter(MicroAgent):
    """
    Advanced goal interpretation system that analyzes conversations
    to extract actionable objectives and requirements.
    """
    
    def __init__(self):
        super().__init__(
            name="Goal Interpreter",
            description="Advanced goal interpretation system that analyzes conversations to extract actionable objectives and requirements."
        )
        self.version = "2.0.0"
        self.capabilities = [
            "natural_language_processing",
            "intent_recognition",
            "requirement_extraction",
            "conversation_analysis"
        ]
        
        # Intent patterns for different types of goals
        self.intent_patterns = {
            'research': [
                r'understand.*landscape',
                r'analyze.*market',
                r'research.*competitors?',
                r'investigate.*trends?',
                r'study.*industry'
            ],
            'automation': [
                r'automate.*process',
                r'streamline.*workflow',
                r'optimize.*operations',
                r'reduce.*manual',
                r'eliminate.*repetitive'
            ],
            'analysis': [
                r'analyze.*data',
                r'process.*information',
                r'examine.*patterns',
                r'evaluate.*performance',
                r'assess.*quality'
            ],
            'creation': [
                r'create.*system',
                r'build.*application',
                r'develop.*solution',
                r'generate.*content',
                r'design.*interface'
            ]
        }
        
        # Constraint indicators
        self.constraint_patterns = [
            r'budget.*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'deadline.*(\d{1,2}\/\d{1,2}\/\d{4}|\d+ (?:days?|weeks?|months?))',
            r'requires?.*(?:compliance|regulation|standard)',
            r'must.*(?:include|exclude|avoid|support)',
            r'limited.*(?:resources|time|access)'
        ]
        
    def _process(self, input_data: Dict[str, Any]) -> GoalInterpreterOutput:
        """Extract and structure goals from conversation."""
        # Convert dictionary to strongly typed input
        typed_input = GoalInterpreterInput(**input_data)
        conversation = typed_input.conversation_history
        context = typed_input.context or {}
        
        # 1. Identify primary objective
        primary_objective = self._extract_primary_objective(conversation)
        
        # 2. Extract constraints
        constraints = self._extract_constraints(conversation)
        
        # 3. Identify success criteria
        success_criteria = self._extract_success_criteria(conversation)
        
        # 4. Determine preferred solutions
        preferred_solutions = self._extract_preferred_solutions(conversation)
        
        # 5. Assess complexity and urgency
        complexity_score = self._calculate_complexity(conversation)
        urgency_level = self._assess_urgency(conversation)
        
        # 6. Identify required capabilities
        required_capabilities = self._identify_capabilities(conversation)
        
        # 7. Estimate duration
        estimated_duration = self._estimate_duration(conversation, complexity_score)
        
        # 8. Generate clarification questions
        clarification_questions = self._generate_clarifications(
            primary_objective, constraints, success_criteria
        )
        
        # 9. Recommend agents
        recommended_agents = self._recommend_agents(required_capabilities)
        
        # 10. Suggest execution strategy
        execution_strategy = self._suggest_execution_strategy(
            complexity_score, urgency_level, required_capabilities
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            primary_objective, constraints, success_criteria
        )
        
        extracted_goal = ExtractedGoal(
            primary_objective=primary_objective,
            key_constraints=constraints,
            success_criteria=success_criteria,
            preferred_solutions=preferred_solutions,
            urgency_level=urgency_level,
            complexity_score=complexity_score,
            required_capabilities=required_capabilities,
            estimated_duration=estimated_duration
        )
        
        return GoalInterpreterOutput(
            extracted_goal=extracted_goal.__dict__,
            confidence_score=confidence_score,
            clarification_questions=clarification_questions,
            recommended_agents=recommended_agents,
            execution_strategy=execution_strategy
        )
    
    def _extract_primary_objective(self, conversation: str) -> str:
        """Extract the main objective from conversation."""
        # Look for explicit goal statements
        goal_indicators = [
            r'I need to (.+?)(?:\.|$)',
            r'I want to (.+?)(?:\.|$)',
            r'Help me (.+?)(?:\.|$)',
            r'My goal is to (.+?)(?:\.|$)',
            r'The objective is (.+?)(?:\.|$)'
        ]
        
        for pattern in goal_indicators:
            match = re.search(pattern, conversation, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: extract based on intent patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, conversation, re.IGNORECASE):
                    return f"{intent.title()} objective identified in conversation"
        
        return "Objective requires clarification"
    
    def _extract_constraints(self, conversation: str) -> List[str]:
        """Extract constraints and limitations."""
        constraints = []
        
        for pattern in self.constraint_patterns:
            matches = re.finditer(pattern, conversation, re.IGNORECASE)
            for match in matches:
                constraints.append(match.group(0))
        
        # Look for additional constraint indicators
        constraint_keywords = [
            'cannot', 'must not', 'avoid', 'limited', 'restricted',
            'compliance', 'regulation', 'policy', 'budget', 'deadline'
        ]
        
        sentences = conversation.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in constraint_keywords):
                constraints.append(sentence.strip())
        
        return constraints
    
    def _extract_success_criteria(self, conversation: str) -> List[str]:
        """Extract success criteria and expected outcomes."""
        criteria = []
        
        success_indicators = [
            r'success.*when (.+?)(?:\.|$)',
            r'completed.*if (.+?)(?:\.|$)',
            r'expect.*to (.+?)(?:\.|$)',
            r'result.*should (.+?)(?:\.|$)',
            r'outcome.*is (.+?)(?:\.|$)'
        ]
        
        for pattern in success_indicators:
            matches = re.finditer(pattern, conversation, re.IGNORECASE)
            for match in matches:
                criteria.append(match.group(1).strip())
        
        return criteria if criteria else ["Success criteria require definition"]
    
    def _extract_preferred_solutions(self, conversation: str) -> List[str]:
        """Extract preferred approaches or solutions."""
        solutions = []
        
        preference_patterns = [
            r'prefer.*(?:using|with) (.+?)(?:\.|$)',
            r'like.*to use (.+?)(?:\.|$)',
            r'should.*leverage (.+?)(?:\.|$)',
            r'using (.+?) would be ideal'
        ]
        
        for pattern in preference_patterns:
            matches = re.finditer(pattern, conversation, re.IGNORECASE)
            for match in matches:
                solutions.append(match.group(1).strip())
        
        return solutions
    
    def _calculate_complexity(self, conversation: str) -> float:
        """Calculate complexity score based on conversation content."""
        complexity_indicators = {
            'multiple': 0.2,
            'integrate': 0.3,
            'complex': 0.4,
            'enterprise': 0.3,
            'scalable': 0.2,
            'real-time': 0.3,
            'machine learning': 0.4,
            'ai': 0.3,
            'distributed': 0.3,
            'compliance': 0.2
        }
        
        score = 0.0
        word_count = len(conversation.split())
        
        # Base complexity from length
        if word_count > 500:
            score += 0.3
        elif word_count > 200:
            score += 0.2
        elif word_count > 100:
            score += 0.1
        
        # Add complexity indicators
        for indicator, weight in complexity_indicators.items():
            if indicator in conversation.lower():
                score += weight
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _assess_urgency(self, conversation: str) -> int:
        """Assess urgency level (1-10 scale)."""
        urgency_keywords = {
            'immediate': 10,
            'urgent': 9,
            'asap': 9,
            'critical': 8,
            'priority': 7,
            'soon': 6,
            'quickly': 5,
            'eventually': 3,
            'when possible': 2
        }
        
        for keyword, level in urgency_keywords.items():
            if keyword in conversation.lower():
                return level
        
        # Check for deadline indicators
        if re.search(r'\d+ days?', conversation.lower()):
            return 8
        elif re.search(r'\d+ weeks?', conversation.lower()):
            return 6
        elif re.search(r'\d+ months?', conversation.lower()):
            return 4
        
        return 5  # Default medium urgency
    
    def _identify_capabilities(self, conversation: str) -> List[str]:
        """Identify required agent capabilities."""
        capability_mapping = {
            'data': ['data_processing', 'data_analysis'],
            'web': ['web_scraping', 'api_integration'],
            'analyze': ['analysis', 'pattern_recognition'],
            'security': ['security_analysis', 'vulnerability_assessment'],
            'report': ['report_generation', 'visualization'],
            'automate': ['automation', 'workflow_management'],
            'monitor': ['monitoring', 'alerting'],
            'research': ['research', 'information_gathering']
        }
        
        capabilities = set()
        conversation_lower = conversation.lower()
        
        for keyword, caps in capability_mapping.items():
            if keyword in conversation_lower:
                capabilities.update(caps)
        
        return list(capabilities)
    
    def _estimate_duration(self, conversation: str, complexity: float) -> str:
        """Estimate project duration."""
        # Extract explicit duration mentions
        duration_patterns = [
            r'(\d+)\s*(?:days?|weeks?|months?)',
            r'by\s+(\w+\s+\d+)',
            r'deadline.*(\d{1,2}\/\d{1,2}\/\d{4})'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, conversation, re.IGNORECASE)
            if match:
                return f"Explicit timeline: {match.group(0)}"
        
        # Estimate based on complexity
        if complexity < 0.3:
            return "1-3 days"
        elif complexity < 0.6:
            return "1-2 weeks"
        elif complexity < 0.8:
            return "2-4 weeks"
        else:
            return "1-2 months"
    
    def _generate_clarifications(self, objective: str, constraints: List[str], 
                               success_criteria: List[str]) -> List[str]:
        """Generate clarification questions."""
        questions = []
        
        if objective == "Objective requires clarification":
            questions.append("Could you clarify your primary objective?")
        
        if not constraints:
            questions.extend([
                "Are there any budget constraints?",
                "What is your preferred timeline?",
                "Are there any compliance requirements?"
            ])
        
        if not success_criteria or success_criteria == ["Success criteria require definition"]:
            questions.extend([
                "How will you measure success?",
                "What specific outcomes are you expecting?"
            ])
        
        return questions
    
    def _recommend_agents(self, capabilities: List[str]) -> List[str]:
        """Recommend agents based on required capabilities."""
        agent_mapping = {
            'data_processing': ['DataProcessor', 'ETLAgent'],
            'web_scraping': ['WebScraper', 'SurfaceWebScraper'],
            'analysis': ['DataAnalyzer', 'StaticAnalyzer'],
            'security_analysis': ['SecurityAuditor', 'VulnerabilityScanner'],
            'report_generation': ['ReportGenerator', 'SummaryGenerator'],
            'automation': ['WorkflowManager', 'TaskScheduler'],
            'monitoring': ['TelemetryCollector', 'AnomalyDetector']
        }
        
        recommended = set()
        for capability in capabilities:
            if capability in agent_mapping:
                recommended.update(agent_mapping[capability])
        
        return list(recommended)
    
    def _suggest_execution_strategy(self, complexity: float, urgency: int, 
                                  capabilities: List[str]) -> str:
        """Suggest execution strategy."""
        if urgency >= 8 and complexity < 0.5:
            return "Fast-track: Single agent with immediate execution"
        elif complexity >= 0.7:
            return "Parallel: Multi-agent team with orchestrated workflow"
        elif len(capabilities) > 3:
            return "Sequential: Pipeline approach with staged execution"
        else:
            return "Standard: Balanced approach with monitoring"
    
    def _calculate_confidence(self, objective: str, constraints: List[str], 
                            success_criteria: List[str]) -> float:
        """Calculate confidence in goal extraction."""
        confidence = 0.5  # Base confidence
        
        if objective != "Objective requires clarification":
            confidence += 0.3
        
        if constraints:
            confidence += 0.1
        
        if success_criteria and success_criteria != ["Success criteria require definition"]:
            confidence += 0.1
        
        return min(confidence, 1.0)
