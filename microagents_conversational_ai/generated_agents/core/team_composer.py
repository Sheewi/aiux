"""
Team Composer - Dynamic Team Formation System
Automatically assembles optimal agent teams for complex objectives.
"""

import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from core.base_agent import MicroAgent, HybridAgent
from core.registry import AgentRegistry
from pydantic import BaseModel
import itertools

@dataclass
class AgentCapability:
    """Represents an agent's capability with scoring."""
    name: str
    strength: float  # 0-1 scale
    dependencies: List[str]
    resource_cost: float
    execution_time: float

@dataclass
class TeamMember:
    """Represents a team member with role assignment."""
    agent_name: str
    agent_class: type
    role: str
    capabilities: List[AgentCapability]
    priority: int
    dependencies: List[str]

@dataclass
class TeamComposition:
    """Complete team structure with optimization metrics."""
    members: List[TeamMember]
    execution_graph: nx.DiGraph
    estimated_duration: float
    resource_cost: float
    success_probability: float
    bottlenecks: List[str]
    redundancy_level: float

class TeamComposerInput(BaseModel):
    objective: Dict[str, Any]
    required_capabilities: List[str]
    constraints: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}

class TeamComposerOutput(BaseModel):
    team_composition: Dict[str, Any]
    execution_plan: Dict[str, Any]
    alternative_teams: List[Dict[str, Any]]
    optimization_metrics: Dict[str, float]

class TeamComposer(HybridAgent):
    """
    Advanced team composition system that assembles optimal agent teams
    based on objectives, capabilities, and constraints.
    """
    
    def __init__(self):
        super().__init__(
            name="Team Composer",
            description="Advanced team composition system that assembles optimal agent teams based on objectives, capabilities, and constraints.",
            component_agents=[]
        )
        self.name = "Team Composer"
        self.version = "2.0.0"
        self.registry = AgentRegistry()
        
        # Capability scoring matrix
        self.capability_matrix = {
            'data_processing': {
                'DataProcessor': 0.9,
                'ETLAgent': 0.8,
                'DataAnalyzer': 0.7,
                'StaticAnalyzer': 0.6
            },
            'web_scraping': {
                'WebScraper': 0.9,
                'SurfaceWebScraper': 0.8,
                'APIIntegrator': 0.7,
                'DataCollector': 0.6
            },
            'security_analysis': {
                'SecurityAuditor': 0.9,
                'VulnerabilityScanner': 0.8,
                'ThreatDetector': 0.7,
                'PenetrationTester': 0.6
            },
            'analysis': {
                'DataAnalyzer': 0.9,
                'StaticAnalyzer': 0.8,
                'SentimentAnalyzer': 0.7,
                'MarketAnalyst': 0.6
            },
            'reporting': {
                'ReportGenerator': 0.9,
                'SummaryGenerator': 0.8,
                'Visualizer': 0.7,
                'PresentationBuilder': 0.6
            },
            'orchestration': {
                'WorkflowManager': 0.9,
                'TaskScheduler': 0.8,
                'Orchestrator': 0.7,
                'TaskDecomposer': 0.6
            }
        }
        
        # Role definitions
        self.role_definitions = {
            'lead': 'Primary coordination and decision making',
            'collector': 'Data gathering and initial processing',
            'processor': 'Core data processing and transformation',
            'analyzer': 'Analysis and pattern recognition',
            'validator': 'Quality assurance and validation',
            'reporter': 'Output generation and reporting',
            'monitor': 'Progress tracking and health monitoring'
        }
        
        # Synergy bonuses for agent combinations
        self.synergy_matrix = {
            ('WebScraper', 'DataProcessor'): 0.15,
            ('SecurityAuditor', 'VulnerabilityScanner'): 0.20,
            ('DataAnalyzer', 'Visualizer'): 0.12,
            ('WorkflowManager', 'TaskScheduler'): 0.18,
            ('ThreatDetector', 'SecurityMonitor'): 0.14
        }
    
    def _process(self, input_data: Dict[str, Any]) -> TeamComposerOutput:
        """Compose optimal team for the given objective."""
        # Convert the dictionary to a TeamComposerInput object
        team_input = TeamComposerInput(**input_data)
        
        objective = team_input.objective
        required_capabilities = team_input.required_capabilities
        constraints = team_input.constraints
        preferences = team_input.preferences
        
        # 1. Analyze objective complexity
        complexity_score = self._analyze_complexity(objective)
        
        # 2. Map capabilities to available agents
        candidate_agents = self._map_capabilities_to_agents(required_capabilities)
        
        # 3. Generate team configurations
        team_configurations = self._generate_team_configurations(
            candidate_agents, complexity_score, constraints
        )
        
        # 4. Optimize team compositions
        optimized_teams = self._optimize_teams(
            team_configurations, objective, constraints, preferences
        )
        
        # 5. Select best team
        best_team = optimized_teams[0] if optimized_teams else None
        
        # 6. Create execution plan
        execution_plan = self._create_execution_plan(best_team) if best_team else {}
        
        # 7. Calculate optimization metrics
        optimization_metrics = self._calculate_optimization_metrics(best_team) if best_team else {}
        
        return TeamComposerOutput(
            team_composition=best_team.__dict__ if best_team else {},
            execution_plan=execution_plan,
            alternative_teams=[team.__dict__ for team in optimized_teams[1:6]],  # Top 5 alternatives
            optimization_metrics=optimization_metrics
        )
    
    def _analyze_complexity(self, objective: Dict[str, Any]) -> float:
        """Analyze objective complexity to determine team size and structure."""
        complexity = 0.0
        
        # Base complexity from objective scope
        primary_obj = objective.get('primary_objective', '')
        if 'enterprise' in primary_obj.lower():
            complexity += 0.3
        if 'real-time' in primary_obj.lower():
            complexity += 0.2
        if 'distributed' in primary_obj.lower():
            complexity += 0.25
        
        # Constraint complexity
        constraints = objective.get('key_constraints', [])
        complexity += len(constraints) * 0.05
        
        # Capability diversity
        capabilities = objective.get('required_capabilities', [])
        complexity += len(capabilities) * 0.03
        
        return min(complexity, 1.0)
    
    def _map_capabilities_to_agents(self, capabilities: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Map required capabilities to available agents with scores."""
        capability_agents = {}
        
        for capability in capabilities:
            agents = []
            if capability in self.capability_matrix:
                for agent_name, score in self.capability_matrix[capability].items():
                    # Check if agent exists in registry
                    if self.registry.get_agent(agent_name):
                        agents.append((agent_name, score))
            
            # Sort by capability score
            agents.sort(key=lambda x: x[1], reverse=True)
            capability_agents[capability] = agents
        
        return capability_agents
    
    def _generate_team_configurations(self, candidate_agents: Dict[str, List[Tuple[str, float]]], 
                                    complexity: float, constraints: Dict[str, Any]) -> List[List[TeamMember]]:
        """Generate multiple team configuration options."""
        configurations = []
        
        # Determine team size based on complexity
        if complexity < 0.3:
            team_sizes = [2, 3]
        elif complexity < 0.6:
            team_sizes = [3, 4, 5]
        else:
            team_sizes = [4, 5, 6, 7]
        
        budget_limit = constraints.get('budget', float('inf'))
        time_limit = constraints.get('timeline_days', float('inf'))
        
        for team_size in team_sizes:
            # Generate configurations for this size
            configs = self._generate_size_specific_configs(
                candidate_agents, team_size, budget_limit, time_limit
            )
            configurations.extend(configs)
        
        return configurations[:20]  # Limit to top 20 configurations
    
    def _generate_size_specific_configs(self, candidate_agents: Dict[str, List[Tuple[str, float]]], 
                                      team_size: int, budget_limit: float, 
                                      time_limit: float) -> List[List[TeamMember]]:
        """Generate team configurations for a specific size."""
        configurations = []
        
        # Get top agents for each capability
        capability_agents = {}
        for capability, agents in candidate_agents.items():
            capability_agents[capability] = agents[:min(3, len(agents))]  # Top 3 per capability
        
        # Generate combinations
        capability_names = list(capability_agents.keys())
        
        # For small teams, one agent might handle multiple capabilities
        if team_size < len(capability_names):
            # Select most critical capabilities
            critical_caps = capability_names[:team_size]
            for combo in itertools.product(*[capability_agents[cap] for cap in critical_caps]):
                team = self._create_team_from_combo(combo, critical_caps)
                if self._validate_team_constraints(team, budget_limit, time_limit):
                    configurations.append(team)
        else:
            # Assign agents to capabilities with possible overlap
            for combo in itertools.product(*[capability_agents[cap] for cap in capability_names]):
                team = self._create_team_from_combo(combo, capability_names)
                if len(set(member.agent_name for member in team)) <= team_size:
                    if self._validate_team_constraints(team, budget_limit, time_limit):
                        configurations.append(team)
        
        return configurations[:10]  # Limit per size
    
    def _create_team_from_combo(self, combo: Tuple, capabilities: List[str]) -> List[TeamMember]:
        """Create team members from agent combination."""
        team = []
        agent_roles = {}
        
        for i, (agent_name, score) in enumerate(combo):
            capability = capabilities[i]
            
            # Assign role based on capability and team position
            if agent_name not in agent_roles:
                if i == 0:
                    role = 'lead'
                elif 'collector' in capability.lower() or 'scraper' in capability.lower():
                    role = 'collector'
                elif 'process' in capability.lower():
                    role = 'processor'
                elif 'analy' in capability.lower():
                    role = 'analyzer'
                elif 'report' in capability.lower() or 'visual' in capability.lower():
                    role = 'reporter'
                else:
                    role = 'processor'
                
                agent_roles[agent_name] = role
            
            # Get agent class from registry
            agent_class = self.registry.get_agent(agent_name)
            
            # Skip if agent class not found
            if agent_class is None:
                continue
                
            team_member = TeamMember(
                agent_name=agent_name,
                agent_class=agent_class,
                role=agent_roles[agent_name],
                capabilities=[AgentCapability(
                    name=capability,
                    strength=score,
                    dependencies=[],
                    resource_cost=self._estimate_resource_cost(agent_name),
                    execution_time=self._estimate_execution_time(agent_name)
                )],
                priority=i + 1,
                dependencies=[]
            )
            team.append(team_member)
        
        return team
    
    def _validate_team_constraints(self, team: List[TeamMember], 
                                 budget_limit: float, time_limit: float) -> bool:
        """Validate team against constraints."""
        if not team:
            return False
            
        try:
            total_cost = sum(member.capabilities[0].resource_cost for member in team if member.capabilities)
            max_time = max(member.capabilities[0].execution_time for member in team if member.capabilities)
            
            return total_cost <= budget_limit and max_time <= time_limit
        except (IndexError, ValueError, TypeError):
            return False
    
    def _optimize_teams(self, configurations: List[List[TeamMember]], 
                       objective: Dict[str, Any], constraints: Dict[str, Any], 
                       preferences: Dict[str, Any]) -> List[TeamComposition]:
        """Optimize team configurations using multi-criteria optimization."""
        optimized_teams = []
        
        for config in configurations:
            # Calculate optimization metrics
            execution_graph = self._build_execution_graph(config)
            duration = self._estimate_team_duration(config, execution_graph)
            cost = self._calculate_team_cost(config)
            success_prob = self._calculate_success_probability(config)
            bottlenecks = self._identify_bottlenecks(execution_graph)
            redundancy = self._calculate_redundancy_level(config)
            
            # Apply synergy bonuses
            synergy_bonus = self._calculate_synergy_bonus(config)
            success_prob = min(success_prob + synergy_bonus, 1.0)
            
            team_composition = TeamComposition(
                members=config,
                execution_graph=execution_graph,
                estimated_duration=duration,
                resource_cost=cost,
                success_probability=success_prob,
                bottlenecks=bottlenecks,
                redundancy_level=redundancy
            )
            
            optimized_teams.append(team_composition)
        
        # Sort by composite score
        def composite_score(team: TeamComposition) -> float:
            # Weighted score considering multiple factors
            time_weight = preferences.get('time_priority', 0.3)
            cost_weight = preferences.get('cost_priority', 0.2)
            quality_weight = preferences.get('quality_priority', 0.4)
            redundancy_weight = preferences.get('redundancy_priority', 0.1)
            
            # Normalize metrics
            time_score = 1.0 / (1.0 + team.estimated_duration / 10.0)
            cost_score = 1.0 / (1.0 + team.resource_cost / 1000.0)
            quality_score = team.success_probability
            redundancy_score = team.redundancy_level
            
            return (time_weight * time_score + 
                   cost_weight * cost_score + 
                   quality_weight * quality_score + 
                   redundancy_weight * redundancy_score)
        
        optimized_teams.sort(key=composite_score, reverse=True)
        return optimized_teams
    
    def _build_execution_graph(self, team: List[TeamMember]) -> nx.DiGraph:
        """Build execution dependency graph for the team."""
        graph = nx.DiGraph()
        
        # Add nodes for each team member
        for member in team:
            graph.add_node(member.agent_name, 
                          role=member.role,
                          execution_time=member.capabilities[0].execution_time)
        
        # Add edges based on role dependencies
        role_dependencies = {
            'collector': [],
            'processor': ['collector'],
            'analyzer': ['processor'],
            'validator': ['analyzer'],
            'reporter': ['analyzer', 'validator'],
            'monitor': []  # Can run in parallel
        }
        
        for member in team:
            required_roles = role_dependencies.get(member.role, [])
            for other_member in team:
                if other_member.role in required_roles:
                    graph.add_edge(other_member.agent_name, member.agent_name)
        
        return graph
    
    def _estimate_team_duration(self, team: List[TeamMember], graph: nx.DiGraph) -> float:
        """Estimate total execution duration considering parallelization."""
        if not graph.nodes():
            return 0.0
        
        # Calculate critical path
        try:
            # Add execution times as weights
            for node in graph.nodes():
                try:
                    member = next(m for m in team if m.agent_name == node)
                    graph.nodes[node]['weight'] = member.capabilities[0].execution_time
                except (StopIteration, IndexError):
                    # If member not found or no capabilities, use default
                    graph.nodes[node]['weight'] = 1.0
            
            # Find longest path (critical path)
            if nx.is_directed_acyclic_graph(graph):
                topo_sort = list(nx.topological_sort(graph))
                longest_path = 0.0
                
                for node in topo_sort:
                    node_time = graph.nodes[node]['weight']
                    max_predecessor_time = 0.0
                    
                    for pred in graph.predecessors(node):
                        pred_total = graph.nodes[pred].get('total_time', 0.0)
                        max_predecessor_time = max(max_predecessor_time, pred_total)
                    
                    total_time = max_predecessor_time + node_time
                    graph.nodes[node]['total_time'] = total_time
                    longest_path = max(longest_path, total_time)
                
                return longest_path
            else:
                # Fallback for cyclic graphs
                return sum(member.capabilities[0].execution_time for member in team)
        
        except Exception:
            # Fallback calculation
            return max(member.capabilities[0].execution_time for member in team)
    
    def _calculate_team_cost(self, team: List[TeamMember]) -> float:
        """Calculate total resource cost for the team."""
        if not team:
            return 0.0
        try:
            return sum(member.capabilities[0].resource_cost for member in team if member.capabilities)
        except (IndexError, TypeError, AttributeError):
            return 0.0
    
    def _calculate_success_probability(self, team: List[TeamMember]) -> float:
        """Calculate probability of successful execution."""
        if not team:
            return 0.0
            
        base_prob = 0.8  # Base success probability
        
        try:
            # Factor in individual agent capabilities
            valid_capabilities = [member.capabilities[0].strength for member in team if member.capabilities]
            if not valid_capabilities:
                return 0.5  # Default probability for teams without valid capabilities
                
            avg_capability = sum(valid_capabilities) / len(valid_capabilities)
            capability_bonus = (avg_capability - 0.5) * 0.3
            
            # Factor in team size (too small or too large reduces success)
            size_factor = 1.0
            if len(team) < 2:
                size_factor = 0.8
            elif len(team) > 6:
                size_factor = 0.9 - (len(team) - 6) * 0.05
            
            return min(base_prob + capability_bonus * size_factor, 1.0)
        except (TypeError, ZeroDivisionError, IndexError):
            return 0.5  # Default fallback probability
    
    def _identify_bottlenecks(self, graph: nx.DiGraph) -> List[str]:
        """Identify potential bottlenecks in execution flow."""
        bottlenecks = []
        
        if not graph or not graph.nodes():
            return bottlenecks
        
        try:
            for node in graph.nodes():
                # High in-degree suggests bottleneck
                in_deg = graph.in_degree[node] if hasattr(graph.in_degree, '__getitem__') else graph.in_degree(node)
                if isinstance(in_deg, int) and in_deg > 2:
                    bottlenecks.append(f"High dependency: {node}")
                
                # High out-degree suggests critical path
                out_deg = graph.out_degree[node] if hasattr(graph.out_degree, '__getitem__') else graph.out_degree(node)
                if isinstance(out_deg, int) and out_deg > 2:
                    bottlenecks.append(f"Critical dependency: {node}")
        except (TypeError, KeyError, AttributeError):
            # Fallback: assume no bottlenecks if we can't calculate
            pass
        
        return bottlenecks
    
    def _calculate_redundancy_level(self, team: List[TeamMember]) -> float:
        """Calculate redundancy level for fault tolerance."""
        capability_counts = {}
        
        for member in team:
            for capability in member.capabilities:
                capability_counts[capability.name] = capability_counts.get(capability.name, 0) + 1
        
        if not capability_counts:
            return 0.0
        
        redundant_capabilities = sum(1 for count in capability_counts.values() if count > 1)
        return redundant_capabilities / len(capability_counts)
    
    def _calculate_synergy_bonus(self, team: List[TeamMember]) -> float:
        """Calculate synergy bonus from agent combinations."""
        total_bonus = 0.0
        agent_names = [member.agent_name for member in team]
        
        for i, agent1 in enumerate(agent_names):
            for agent2 in agent_names[i+1:]:
                if (agent1, agent2) in self.synergy_matrix:
                    total_bonus += self.synergy_matrix[(agent1, agent2)]
                elif (agent2, agent1) in self.synergy_matrix:
                    total_bonus += self.synergy_matrix[(agent2, agent1)]
        
        return total_bonus
    
    def _estimate_resource_cost(self, agent_name: str) -> float:
        """Estimate resource cost for an agent."""
        # Base costs by agent type
        base_costs = {
            'DataProcessor': 50.0,
            'WebScraper': 30.0,
            'SecurityAuditor': 80.0,
            'ReportGenerator': 40.0,
            'WorkflowManager': 60.0
        }
        return base_costs.get(agent_name, 45.0)
    
    def _estimate_execution_time(self, agent_name: str) -> float:
        """Estimate execution time for an agent."""
        # Base execution times in hours
        base_times = {
            'DataProcessor': 2.0,
            'WebScraper': 1.5,
            'SecurityAuditor': 4.0,
            'ReportGenerator': 1.0,
            'WorkflowManager': 0.5
        }
        return base_times.get(agent_name, 2.0)
    
    def _create_execution_plan(self, team: TeamComposition) -> Dict[str, Any]:
        """Create detailed execution plan for the team."""
        if not team:
            return {}
        
        phases = []
        
        # Analyze execution graph to create phases
        if team.execution_graph:
            try:
                # Group nodes by execution level
                levels = []
                remaining_nodes = set(team.execution_graph.nodes())
                
                while remaining_nodes:
                    # Find nodes with no dependencies in remaining set
                    current_level = []
                    for node in list(remaining_nodes):
                        predecessors = set(team.execution_graph.predecessors(node))
                        if not predecessors.intersection(remaining_nodes):
                            current_level.append(node)
                    
                    if not current_level:  # Avoid infinite loop
                        current_level = list(remaining_nodes)
                    
                    levels.append(current_level)
                    remaining_nodes -= set(current_level)
                
                # Create phases from levels
                for i, level in enumerate(levels):
                    try:
                        estimated_duration = max(
                            next(m.capabilities[0].execution_time for m in team.members if m.agent_name == agent and m.capabilities)
                            for agent in level
                        )
                    except (ValueError, StopIteration, IndexError):
                        estimated_duration = 1.0  # Default duration
                    
                    phase = {
                        'phase': i + 1,
                        'agents': level,
                        'parallel_execution': len(level) > 1,
                        'estimated_duration': estimated_duration
                    }
                    phases.append(phase)
            
            except Exception:
                # Fallback to sequential execution
                phases = []
                for i, member in enumerate(team.members):
                    try:
                        duration = member.capabilities[0].execution_time if member.capabilities else 1.0
                    except (IndexError, AttributeError):
                        duration = 1.0
                    
                    phases.append({
                        'phase': i + 1,
                        'agents': [member.agent_name],
                        'parallel_execution': False,
                        'estimated_duration': duration
                    })
        
        return {
            'phases': phases,
            'total_estimated_duration': team.estimated_duration,
            'resource_requirements': team.resource_cost,
            'success_probability': team.success_probability,
            'critical_path': phases,
            'monitoring_points': [f"After phase {i+1}" for i in range(len(phases))]
        }
    
    def _calculate_optimization_metrics(self, team: TeamComposition) -> Dict[str, float]:
        """Calculate detailed optimization metrics."""
        if not team or not team.members:
            return {}
        
        try:
            execution_plan = self._create_execution_plan(team)
            phases = execution_plan.get('phases', [])
            
            # Safe calculation of parallelization factor
            parallel_phases = len([p for p in phases if p.get('parallel_execution', False)])
            total_phases = len(phases)
            parallelization_factor = parallel_phases / max(total_phases, 1)
            
            # Safe calculation of average agent strength
            valid_strengths = [
                member.capabilities[0].strength 
                for member in team.members 
                if member.capabilities
            ]
            avg_strength = sum(valid_strengths) / len(valid_strengths) if valid_strengths else 0.0
            
            return {
                'team_size': len(team.members),
                'execution_efficiency': 1.0 / team.estimated_duration if team.estimated_duration > 0 else 0,
                'cost_efficiency': team.success_probability / team.resource_cost if team.resource_cost > 0 else 0,
                'redundancy_level': team.redundancy_level,
                'parallelization_factor': parallelization_factor,
                'bottleneck_risk': len(team.bottlenecks) / len(team.members) if team.members else 0,
                'capability_coverage': len(set(cap.name for member in team.members for cap in member.capabilities if member.capabilities)),
                'average_agent_strength': avg_strength
            }
        except (AttributeError, TypeError, ZeroDivisionError):
            return {
                'team_size': len(team.members) if team.members else 0,
                'execution_efficiency': 0.0,
                'cost_efficiency': 0.0,
                'redundancy_level': 0.0,
                'parallelization_factor': 0.0,
                'bottleneck_risk': 0.0,
                'capability_coverage': 0,
                'average_agent_strength': 0.0
            }
