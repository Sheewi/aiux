"""
Market Research Orchestrator - Complete Example Implementation
Demonstrates the conversational AI system for market research projects.
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from core.goal_interpreter import GoalInterpreter
from core.team_composer import TeamComposer
from core.live_orchestrator import LiveOrchestrator
from core.progress_visualizer import ProgressVisualizer

class MarketResearchOrchestrator:
    """
    Complete example implementation showing how the enhanced microagent
    system handles a complex market research project from conversation to completion.
    """
    
    def __init__(self):
        # Initialize core components
        self.goal_interpreter = GoalInterpreter()
        self.team_composer = TeamComposer()
        self.live_orchestrator = LiveOrchestrator()
        self.progress_visualizer = ProgressVisualizer()
        
        # Conversation history and state
        self.conversation_history = []
        self.current_project = None
        
    def process_user_request(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and orchestrate the complete workflow.
        
        Example: "I need to understand the competitive landscape for electric scooters in Europe"
        """
        
        # 1. Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'speaker': 'user',
            'message': user_input
        })
        
        # 2. Interpret the goal using natural language processing
        conversation_text = self._format_conversation_history()
        
        goal_input = type('GoalInput', (), {
            'conversation_history': conversation_text,
            'context': {'domain': 'market_research'},
            'user_preferences': {'urgency': 'medium', 'detail_level': 'comprehensive'}
        })()
        
        goal_result = self.goal_interpreter._process(goal_input)
        
        # 3. Check if clarification is needed
        if goal_result.clarification_questions:
            return {
                'status': 'clarification_needed',
                'questions': goal_result.clarification_questions,
                'confidence': goal_result.confidence_score,
                'preliminary_plan': goal_result.execution_strategy
            }
        
        # 4. Compose optimal team for the research
        team_input = type('TeamInput', (), {
            'objective': goal_result.extracted_goal,
            'required_capabilities': [
                'web_scraping', 'data_analysis', 'market_research', 
                'competitive_analysis', 'report_generation'
            ],
            'constraints': {
                'budget': 5000,
                'timeline_days': 14
            },
            'preferences': {
                'quality_priority': 0.4,
                'speed_priority': 0.3,
                'cost_priority': 0.3
            }
        })()
        
        team_result = self.team_composer._process(team_input)
        
        # 5. Start live orchestration
        orchestrator_input = type('OrchestratorInput', (), {
            'goal_specification': goal_result.extracted_goal,
            'execution_preferences': {
                'monitoring_level': 'detailed',
                'adaptation_enabled': True,
                'real_time_updates': True
            },
            'monitoring_config': {
                'health_check_interval': 30,
                'progress_updates': True
            }
        })()
        
        workflow_result = self.live_orchestrator._process(orchestrator_input)
        
        # 6. Create visualization dashboard
        dashboard_input = type('DashboardInput', (), {
            'workflow_id': workflow_result.workflow_id,
            'dashboard_type': 'executive',
            'update_interval': 10,
            'custom_components': [
                {
                    'name': 'competitive_analysis',
                    'type': 'chart',
                    'title': 'Competitive Positioning',
                    'config': {'chart_type': 'scatter'}
                },
                {
                    'name': 'market_trends',
                    'type': 'chart',
                    'title': 'Market Trends',
                    'config': {'chart_type': 'line'}
                }
            ]
        })()
        
        dashboard_result = self.progress_visualizer._process(dashboard_input)
        
        # 7. Store project information
        self.current_project = {
            'project_id': workflow_result.workflow_id,
            'goal': goal_result.extracted_goal,
            'team': team_result.team_composition,
            'workflow': workflow_result,
            'dashboard': dashboard_result,
            'started_at': datetime.now().isoformat()
        }
        
        # 8. Return comprehensive response
        return {
            'status': 'project_initiated',
            'project_id': workflow_result.workflow_id,
            'goal_analysis': {
                'primary_objective': goal_result.extracted_goal['primary_objective'],
                'confidence': goal_result.confidence_score,
                'complexity': goal_result.extracted_goal['complexity_score'],
                'estimated_duration': goal_result.extracted_goal['estimated_duration']
            },
            'team_composition': {
                'team_size': len(team_result.team_composition.get('members', [])),
                'execution_strategy': team_result.execution_plan,
                'success_probability': team_result.optimization_metrics.get('average_agent_strength', 0)
            },
            'workflow_info': {
                'workflow_id': workflow_result.workflow_id,
                'current_state': workflow_result.current_state,
                'progress': workflow_result.progress_percentage
            },
            'dashboard_url': dashboard_result.dashboard_url,
            'next_steps': [
                'Monitor progress via dashboard',
                'Receive automated updates',
                'Review preliminary findings',
                'Provide feedback for adaptations'
            ]
        }
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for goal interpretation."""
        formatted = []
        for entry in self.conversation_history:
            speaker = entry['speaker'].upper()
            message = entry['message']
            timestamp = entry['timestamp']
            formatted.append(f"[{timestamp}] {speaker}: {message}")
        
        return "\n".join(formatted)
    
    def get_project_status(self, project_id: str = None) -> Dict[str, Any]:
        """Get current project status with detailed metrics."""
        if not project_id and self.current_project:
            project_id = self.current_project['project_id']
        
        if not project_id:
            return {'error': 'No active project found'}
        
        # Get workflow status
        workflow_status = self.live_orchestrator.get_workflow_status(project_id)
        
        # Get dashboard info
        dashboard_info = None
        if self.current_project and 'dashboard' in self.current_project:
            dashboard_id = self.current_project['dashboard'].dashboard_url.split('/')[-1]
            dashboard_info = self.progress_visualizer.get_dashboard_info(dashboard_id)
        
        return {
            'project_id': project_id,
            'workflow_status': workflow_status,
            'dashboard_info': dashboard_info,
            'project_metadata': self.current_project
        }
    
    def provide_feedback(self, feedback: str, adaptation_request: str = None) -> Dict[str, Any]:
        """
        Allow user to provide feedback and request adaptations.
        
        Example: "The analysis seems too broad, focus more on premium segment"
        """
        
        if not self.current_project:
            return {'error': 'No active project to provide feedback for'}
        
        # Add feedback to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'speaker': 'user',
            'message': f"FEEDBACK: {feedback}"
        })
        
        if adaptation_request:
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'speaker': 'user',
                'message': f"ADAPTATION REQUEST: {adaptation_request}"
            })
        
        # Process feedback through goal interpreter
        conversation_text = self._format_conversation_history()
        
        goal_input = type('GoalInput', (), {
            'conversation_history': conversation_text,
            'context': {
                'domain': 'market_research',
                'project_id': self.current_project['project_id'],
                'current_goal': self.current_project['goal']
            },
            'user_preferences': {'feedback_mode': True}
        })()
        
        updated_goal = self.goal_interpreter._process(goal_input)
        
        # Check if significant changes are needed
        if updated_goal.confidence_score > 0.7:
            # Trigger workflow adaptation
            project_id = self.current_project['project_id']
            
            # Update workflow with new requirements
            # (In real implementation, would call orchestrator adaptation methods)
            
            return {
                'status': 'feedback_processed',
                'adaptations_applied': True,
                'updated_goal': updated_goal.extracted_goal,
                'message': 'Project adapted based on your feedback. Monitoring for improved results.'
            }
        else:
            return {
                'status': 'feedback_received',
                'adaptations_applied': False,
                'message': 'Feedback noted. Current approach appears aligned with objectives.',
                'clarification_questions': updated_goal.clarification_questions
            }
    
    def generate_interim_report(self, report_type: str = 'executive') -> Dict[str, Any]:
        """Generate interim progress report."""
        
        if not self.current_project:
            return {'error': 'No active project for reporting'}
        
        project_id = self.current_project['project_id']
        workflow_status = self.live_orchestrator.get_workflow_status(project_id)
        
        # Simulate report generation based on current progress
        report_data = {
            'report_type': report_type,
            'project_id': project_id,
            'generated_at': datetime.now().isoformat(),
            'project_overview': {
                'objective': self.current_project['goal']['primary_objective'],
                'status': workflow_status['state'],
                'progress': workflow_status['metrics']['progress_percentage'],
                'started': self.current_project['started_at']
            },
            'key_findings': self._generate_key_findings(workflow_status),
            'competitive_analysis': self._generate_competitive_analysis(),
            'market_insights': self._generate_market_insights(),
            'risk_assessment': self._generate_risk_assessment(workflow_status),
            'recommendations': self._generate_recommendations(),
            'next_steps': self._generate_next_steps(workflow_status)
        }
        
        return {
            'status': 'report_generated',
            'report': report_data,
            'download_url': f"/reports/{project_id}/interim_{report_type}_{int(datetime.now().timestamp())}.pdf"
        }
    
    def _generate_key_findings(self, workflow_status: Dict[str, Any]) -> List[str]:
        """Generate key findings based on current progress."""
        return [
            "European electric scooter market shows 45% YoY growth",
            "Premium segment (€800+) dominated by 3 main players",
            "Regulatory landscape varies significantly across EU countries",
            "Consumer preference shifting towards longer-range models",
            "Urban mobility policies driving adoption in major cities"
        ]
    
    def _generate_competitive_analysis(self) -> Dict[str, Any]:
        """Generate competitive analysis summary."""
        return {
            'market_leaders': [
                {'company': 'Xiaomi', 'market_share': 28, 'key_strength': 'Price-performance ratio'},
                {'company': 'Segway-Ninebot', 'market_share': 22, 'key_strength': 'Brand recognition'},
                {'company': 'Bird', 'market_share': 15, 'key_strength': 'Sharing platform integration'}
            ],
            'emerging_players': [
                {'company': 'Tier Mobility', 'growth_rate': 85, 'focus': 'Micro-mobility ecosystem'},
                {'company': 'Voi Technology', 'growth_rate': 72, 'focus': 'Sustainable transport'}
            ],
            'competitive_gaps': [
                'Long-range premium models (>50km)',
                'Integration with public transport systems',
                'Advanced safety features'
            ]
        }
    
    def _generate_market_insights(self) -> Dict[str, Any]:
        """Generate market insights summary."""
        return {
            'market_size': {
                'current': '€2.1B (2024)',
                'projected': '€4.8B (2027)',
                'cagr': '31%'
            },
            'regional_breakdown': {
                'western_europe': 65,
                'eastern_europe': 25,
                'nordic': 10
            },
            'key_trends': [
                'Shift towards subscription models',
                'Integration with smart city infrastructure',
                'Focus on battery swapping technology',
                'Regulatory standardization across EU'
            ],
            'user_demographics': {
                'primary_age_group': '25-35',
                'income_level': 'Middle to upper-middle class',
                'usage_pattern': 'Last-mile commuting'
            }
        }
    
    def _generate_risk_assessment(self, workflow_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment based on current analysis."""
        return {
            'market_risks': [
                {'risk': 'Regulatory changes', 'probability': 'Medium', 'impact': 'High'},
                {'risk': 'Economic downturn', 'probability': 'Low', 'impact': 'Medium'},
                {'risk': 'Technology disruption', 'probability': 'High', 'impact': 'Medium'}
            ],
            'competitive_risks': [
                {'risk': 'New market entrants', 'probability': 'High', 'impact': 'Medium'},
                {'risk': 'Price wars', 'probability': 'Medium', 'impact': 'High'}
            ],
            'operational_risks': [
                {'risk': 'Supply chain disruption', 'probability': 'Medium', 'impact': 'High'},
                {'risk': 'Battery technology limitations', 'probability': 'Low', 'impact': 'High'}
            ]
        }
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate strategic recommendations."""
        return [
            {
                'recommendation': 'Focus on premium long-range segment',
                'rationale': 'Underserved market with high growth potential',
                'priority': 'High',
                'timeline': '6-12 months'
            },
            {
                'recommendation': 'Develop partnerships with public transport',
                'rationale': 'Integration opportunities in smart cities',
                'priority': 'Medium',
                'timeline': '12-18 months'
            },
            {
                'recommendation': 'Invest in battery swapping infrastructure',
                'rationale': 'Addresses key user pain point of charging',
                'priority': 'Medium',
                'timeline': '18-24 months'
            }
        ]
    
    def _generate_next_steps(self, workflow_status: Dict[str, Any]) -> List[str]:
        """Generate next steps based on current progress."""
        next_steps = []
        
        if workflow_status['metrics']['progress_percentage'] < 50:
            next_steps.extend([
                'Complete data collection from remaining sources',
                'Conduct deep-dive analysis on key competitors',
                'Validate preliminary findings with industry experts'
            ])
        else:
            next_steps.extend([
                'Finalize comprehensive competitive analysis',
                'Develop market entry strategy recommendations',
                'Prepare final presentation for stakeholders'
            ])
        
        return next_steps

# Example usage demonstration
def demonstrate_market_research_workflow():
    """
    Demonstrate the complete market research workflow from
    user input to final deliverables.
    """
    
    # Initialize the orchestrator
    orchestrator = MarketResearchOrchestrator()
    
    print("=== CONVERSATIONAL AI MARKET RESEARCH SYSTEM ===\n")
    
    # 1. User input
    user_request = "I need to understand the competitive landscape for electric scooters in Europe"
    print(f"User Request: {user_request}\n")
    
    # 2. Process request
    initial_response = orchestrator.process_user_request(user_request)
    print("System Response:")
    print(json.dumps(initial_response, indent=2))
    print("\n" + "="*60 + "\n")
    
    # 3. Simulate project progress check
    print("Checking project status after 30 minutes...\n")
    status = orchestrator.get_project_status()
    print("Project Status:")
    print(json.dumps(status['workflow_status']['metrics'], indent=2))
    print("\n" + "="*60 + "\n")
    
    # 4. User provides feedback
    feedback = "The analysis seems too broad, focus more on the premium segment above €800"
    print(f"User Feedback: {feedback}\n")
    
    feedback_response = orchestrator.provide_feedback(feedback)
    print("Feedback Response:")
    print(json.dumps(feedback_response, indent=2))
    print("\n" + "="*60 + "\n")
    
    # 5. Generate interim report
    print("Generating interim executive report...\n")
    report = orchestrator.generate_interim_report('executive')
    print("Interim Report Summary:")
    print(f"Key Findings: {len(report['report']['key_findings'])} insights")
    print(f"Competitive Analysis: {len(report['report']['competitive_analysis']['market_leaders'])} leaders identified")
    print(f"Recommendations: {len(report['report']['recommendations'])} strategic recommendations")
    print(f"Full report available at: {report['download_url']}")
    
    return orchestrator

if __name__ == "__main__":
    # Run the demonstration
    demo_orchestrator = demonstrate_market_research_workflow()
