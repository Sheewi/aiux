"""
Live Orchestrator - Adaptive Workflow Engine
Real-time workflow management with dynamic adaptation capabilities.
"""

import networkx as nx
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from core.base_agent import MicroAgent
from core.registry import AgentRegistry
from core.team_composer import TeamComposer, TeamComposerInput
from pydantic import BaseModel
import json
from datetime import datetime, timedelta

class WorkflowState(Enum):
    """Workflow execution states."""
    AWAITING_GOAL = "awaiting_goal"
    PLANNING = "planning"
    EXECUTING = "executing"
    ADAPTING = "adapting"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class NodeState(Enum):
    """Individual node execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

@dataclass
class WorkflowNode:
    """Represents a single execution node in the workflow."""
    id: str
    agent_name: str
    agent_instance: Any
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = field(default_factory=dict)
    state: NodeState = NodeState.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    execution_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowMetrics:
    """Workflow execution metrics."""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    bottlenecks: List[str] = field(default_factory=list)
    resource_utilization: float = 0.0

class LiveOrchestratorInput(BaseModel):
    goal_specification: Dict[str, Any]
    execution_preferences: Dict[str, Any] = {}
    monitoring_config: Dict[str, Any] = {}

class LiveOrchestratorOutput(BaseModel):
    workflow_id: str
    execution_status: str
    current_state: str
    progress_percentage: float
    metrics: Dict[str, Any]
    next_actions: List[str]

class LiveOrchestrator(MicroAgent):
    """
    Advanced workflow orchestrator with real-time adaptation,
    monitoring, and recovery capabilities.
    """
    
    def __init__(self):
        super().__init__(
            name="Live Orchestrator",
            description="Advanced workflow orchestrator with real-time adaptation, monitoring, and recovery capabilities."
        )
        self.version = "2.0.0"
        
        # Core components
        self.registry = AgentRegistry()
        self.team_composer = TeamComposer()
        
        # Workflow management
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.active_workflows: Dict[str, threading.Thread] = {}
        
        # State management
        self.current_state = WorkflowState.AWAITING_GOAL
        self.workflow_graph = nx.DiGraph()
        self.execution_history: List[Dict[str, Any]] = []
        
        # Monitoring and adaptation
        self.monitoring_thread: Optional[threading.Thread] = None
        self.adaptation_callbacks: List[Callable] = []
        self.health_checks: Dict[str, Callable] = {}
        
        # Configuration
        self.config = {
            'max_concurrent_workflows': 5,
            'health_check_interval': 30,
            'adaptation_threshold': 0.3,
            'failure_threshold': 3,
            'timeout_threshold': 300,  # 5 minutes
            'recovery_strategies': ['retry', 'fallback', 'skip']
        }
        
        # Start monitoring thread
        self._start_monitoring()
    
    def _process(self, input_data: Dict[str, Any]) -> LiveOrchestratorOutput:
        """Process goal and manage workflow execution."""
        # Convert dictionary input to LiveOrchestratorInput for type safety
        typed_input = LiveOrchestratorInput(
            goal_specification=input_data.get('goal_specification', {}),
            execution_preferences=input_data.get('execution_preferences', {}),
            monitoring_config=input_data.get('monitoring_config', {})
        )
        
        goal_spec = typed_input.goal_specification
        preferences = typed_input.execution_preferences
        monitoring_config = typed_input.monitoring_config
        
        # Generate unique workflow ID
        workflow_id = f"workflow_{int(time.time() * 1000)}"
        
        # Update configuration with monitoring preferences
        self.config.update(monitoring_config)
        
        # Initialize workflow
        workflow_info = self._initialize_workflow(workflow_id, goal_spec, preferences)
        
        # Start workflow execution in separate thread
        execution_thread = threading.Thread(
            target=self._execute_workflow,
            args=(workflow_id,),
            daemon=True
        )
        execution_thread.start()
        self.active_workflows[workflow_id] = execution_thread
        
        # Return initial status
        return LiveOrchestratorOutput(
            workflow_id=workflow_id,
            execution_status="started",
            current_state=self.current_state.value,
            progress_percentage=0.0,
            metrics=self._get_workflow_metrics(workflow_id),
            next_actions=["monitor_progress", "check_status"]
        )
    
    def _initialize_workflow(self, workflow_id: str, goal_spec: Dict[str, Any], 
                           preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a new workflow."""
        # Extract goal components
        primary_objective = goal_spec.get('primary_objective', '')
        required_capabilities = goal_spec.get('required_capabilities', [])
        constraints = goal_spec.get('key_constraints', [])
        
        # Compose optimal team
        team_input = {
            'objective': goal_spec,
            'required_capabilities': required_capabilities,
            'constraints': {
                'budget': preferences.get('budget_limit', 1000),
                'timeline_days': preferences.get('timeline_days', 7)
            },
            'preferences': preferences
        }
        
        team_result = self.team_composer._process(TeamComposerInput(**team_input))
        team_composition = team_result.team_composition
        
        # Build workflow graph
        workflow_graph = self._build_workflow_graph(team_composition)
        
        # Create workflow info
        workflow_info = {
            'id': workflow_id,
            'goal_specification': goal_spec,
            'team_composition': team_composition,
            'workflow_graph': workflow_graph,
            'state': WorkflowState.PLANNING,
            'nodes': {},
            'metrics': WorkflowMetrics(start_time=datetime.now()),
            'preferences': preferences,
            'execution_log': []
        }
        
        self.workflows[workflow_id] = workflow_info
        return workflow_info
    
    def _build_workflow_graph(self, team_composition: Dict[str, Any]) -> nx.DiGraph:
        """Build execution graph from team composition."""
        graph = nx.DiGraph()
        
        if not team_composition or 'members' not in team_composition:
            return graph
        
        members = team_composition.get('members', [])
        
        # Add nodes for each team member
        for member in members:
            agent_name = member.get('agent_name', '')
            role = member.get('role', '')
            
            graph.add_node(agent_name, 
                          role=role,
                          priority=member.get('priority', 1),
                          capabilities=member.get('capabilities', []))
        
        # Add edges based on role dependencies and priorities
        role_order = ['lead', 'collector', 'processor', 'analyzer', 'validator', 'reporter']
        
        for i, current_role in enumerate(role_order[:-1]):
            next_role = role_order[i + 1]
            
            current_agents = [m['agent_name'] for m in members if m.get('role') == current_role]
            next_agents = [m['agent_name'] for m in members if m.get('role') == next_role]
            
            # Add dependencies
            for current_agent in current_agents:
                for next_agent in next_agents:
                    graph.add_edge(current_agent, next_agent)
        
        return graph
    
    def _execute_workflow(self, workflow_id: str):
        """Execute workflow in separate thread."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return
        
        try:
            # Update state
            workflow['state'] = WorkflowState.EXECUTING
            self.current_state = WorkflowState.EXECUTING
            
            # Initialize workflow nodes
            self._initialize_workflow_nodes(workflow)
            
            # Execute nodes in dependency order
            execution_order = list(nx.topological_sort(workflow['workflow_graph']))
            
            for agent_name in execution_order:
                # Check if workflow should continue
                if workflow['state'] in [WorkflowState.FAILED, WorkflowState.PAUSED]:
                    break
                
                # Execute node
                success = self._execute_node(workflow_id, agent_name)
                
                if not success:
                    # Handle failure
                    self._handle_node_failure(workflow_id, agent_name)
                    
                    # Check if workflow should continue or fail
                    if not self._should_continue_after_failure(workflow_id):
                        workflow['state'] = WorkflowState.FAILED
                        break
                
                # Update progress
                self._update_workflow_progress(workflow_id)
            
            # Complete workflow if not failed
            if workflow['state'] != WorkflowState.FAILED:
                workflow['state'] = WorkflowState.COMPLETED
                workflow['metrics'].end_time = datetime.now()
            
        except Exception as e:
            self._log_error(workflow_id, f"Workflow execution error: {str(e)}")
            workflow['state'] = WorkflowState.FAILED
        
        finally:
            # Cleanup
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
    
    def _initialize_workflow_nodes(self, workflow: Dict[str, Any]):
        """Initialize nodes for workflow execution."""
        graph = workflow['workflow_graph']
        goal_spec = workflow['goal_specification']
        
        for agent_name in graph.nodes():
            # Get agent instance from registry
            agent_instance = self.registry.get_agent(agent_name)
            
            # Prepare input data based on goal and dependencies
            input_data = self._prepare_node_input(workflow, agent_name)
            
            # Create workflow node
            node = WorkflowNode(
                id=f"{workflow['id']}_{agent_name}",
                agent_name=agent_name,
                agent_instance=agent_instance,
                input_data=input_data,
                dependencies=list(graph.predecessors(agent_name)),
                execution_context={'workflow_id': workflow['id']}
            )
            
            workflow['nodes'][agent_name] = node
    
    def _prepare_node_input(self, workflow: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Prepare input data for a workflow node."""
        goal_spec = workflow['goal_specification']
        
        # Base input from goal specification
        input_data = {
            'agent_id': f"{workflow['id']}_{agent_name}",
            'objective': goal_spec.get('primary_objective', ''),
            'context': goal_spec,
            'workflow_metadata': {
                'workflow_id': workflow['id'],
                'agent_role': workflow['workflow_graph'].nodes[agent_name].get('role', ''),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Add outputs from dependency nodes
        dependencies = workflow['workflow_graph'].predecessors(agent_name)
        dependency_outputs = {}
        
        for dep_agent in dependencies:
            dep_node = workflow['nodes'].get(dep_agent)
            if dep_node and dep_node.output_data:
                dependency_outputs[dep_agent] = dep_node.output_data
        
        if dependency_outputs:
            input_data['dependency_outputs'] = dependency_outputs
        
        return input_data
    
    def _execute_node(self, workflow_id: str, agent_name: str) -> bool:
        """Execute a single workflow node."""
        workflow = self.workflows[workflow_id]
        node = workflow['nodes'][agent_name]
        
        try:
            # Update node state
            node.state = NodeState.RUNNING
            node.start_time = datetime.now()
            
            # Check dependencies are completed
            if not self._dependencies_completed(workflow, agent_name):
                node.state = NodeState.PENDING
                return False
            
            # Update input with latest dependency outputs
            node.input_data = self._prepare_node_input(workflow, agent_name)
            
            # Execute agent
            if node.agent_instance:
                # Create input object for agent
                agent_input = type('AgentInput', (), node.input_data)()
                result = node.agent_instance._process(agent_input)
                
                # Store output
                if hasattr(result, '__dict__'):
                    node.output_data = result.__dict__
                else:
                    node.output_data = {'result': str(result)}
                
                node.state = NodeState.COMPLETED
                node.end_time = datetime.now()
                
                # Log success
                self._log_execution(workflow_id, agent_name, "completed", node.output_data)
                return True
            else:
                raise Exception(f"Agent instance not found: {agent_name}")
        
        except Exception as e:
            # Handle execution error
            node.state = NodeState.FAILED
            node.end_time = datetime.now()
            node.error_message = str(e)
            
            self._log_error(workflow_id, f"Node {agent_name} failed: {str(e)}")
            return False
    
    def _dependencies_completed(self, workflow: Dict[str, Any], agent_name: str) -> bool:
        """Check if all dependencies for a node are completed."""
        dependencies = workflow['workflow_graph'].predecessors(agent_name)
        
        for dep_agent in dependencies:
            dep_node = workflow['nodes'].get(dep_agent)
            if not dep_node or dep_node.state != NodeState.COMPLETED:
                return False
        
        return True
    
    def _handle_node_failure(self, workflow_id: str, agent_name: str):
        """Handle node execution failure with recovery strategies."""
        workflow = self.workflows[workflow_id]
        node = workflow['nodes'][agent_name]
        
        # Try recovery strategies
        for strategy in self.config['recovery_strategies']:
            if self._apply_recovery_strategy(workflow_id, agent_name, strategy):
                break
        else:
            # All recovery strategies failed
            self._log_error(workflow_id, f"All recovery strategies failed for {agent_name}")
    
    def _apply_recovery_strategy(self, workflow_id: str, agent_name: str, strategy: str) -> bool:
        """Apply a specific recovery strategy."""
        workflow = self.workflows[workflow_id]
        node = workflow['nodes'][agent_name]
        
        if strategy == 'retry' and node.retry_count < node.max_retries:
            # Retry execution
            node.retry_count += 1
            node.state = NodeState.RETRYING
            
            self._log_execution(workflow_id, agent_name, f"retrying_attempt_{node.retry_count}")
            
            # Wait before retry (exponential backoff)
            time.sleep(2 ** node.retry_count)
            
            return self._execute_node(workflow_id, agent_name)
        
        elif strategy == 'fallback':
            # Try to find alternative agent
            alternative_agent = self._find_alternative_agent(workflow, agent_name)
            if alternative_agent:
                # Replace failed agent with alternative
                return self._substitute_agent(workflow_id, agent_name, alternative_agent)
        
        elif strategy == 'skip':
            # Skip failed node and continue
            node.state = NodeState.SKIPPED
            self._log_execution(workflow_id, agent_name, "skipped_due_to_failure")
            return True
        
        return False
    
    def _find_alternative_agent(self, workflow: Dict[str, Any], failed_agent: str) -> Optional[str]:
        """Find alternative agent with similar capabilities."""
        failed_node = workflow['nodes'][failed_agent]
        required_capabilities = [cap.get('name', '') for cap in failed_node.agent_instance.capabilities if hasattr(failed_node.agent_instance, 'capabilities')]
        
        # Search registry for agents with similar capabilities
        all_agents = self.registry.list_agents()
        
        for agent_name in all_agents:
            agent_class = self.registry.get_agent(agent_name)
            if agent_name != failed_agent:
                agent_capabilities = getattr(agent_class, 'capabilities', [])
                if any(cap in agent_capabilities for cap in required_capabilities):
                    return agent_name
        
        return None
    
    def _substitute_agent(self, workflow_id: str, failed_agent: str, alternative_agent: str) -> bool:
        """Substitute failed agent with alternative."""
        workflow = self.workflows[workflow_id]
        
        # Get alternative agent instance
        alternative_instance = self.registry.get_agent(alternative_agent)
        if not alternative_instance:
            return False
        
        # Update node
        failed_node = workflow['nodes'][failed_agent]
        failed_node.agent_name = alternative_agent
        failed_node.agent_instance = alternative_instance
        failed_node.state = NodeState.PENDING
        failed_node.error_message = None
        
        self._log_execution(workflow_id, failed_agent, f"substituted_with_{alternative_agent}")
        
        # Execute with alternative agent
        return self._execute_node(workflow_id, failed_agent)
    
    def _should_continue_after_failure(self, workflow_id: str) -> bool:
        """Determine if workflow should continue after node failure."""
        workflow = self.workflows[workflow_id]
        
        # Count failed nodes
        failed_count = sum(1 for node in workflow['nodes'].values() 
                          if node.state == NodeState.FAILED)
        
        total_count = len(workflow['nodes'])
        failure_rate = failed_count / total_count if total_count > 0 else 0
        
        # Continue if failure rate is below threshold
        return failure_rate < self.config['failure_threshold'] / 10  # Convert to percentage
    
    def _update_workflow_progress(self, workflow_id: str):
        """Update workflow progress metrics."""
        workflow = self.workflows[workflow_id]
        nodes = workflow['nodes']
        
        total_nodes = len(nodes)
        completed_nodes = sum(1 for node in nodes.values() 
                            if node.state in [NodeState.COMPLETED, NodeState.SKIPPED])
        failed_nodes = sum(1 for node in nodes.values() 
                         if node.state == NodeState.FAILED)
        
        # Update metrics
        metrics = workflow['metrics']
        metrics.total_nodes = total_nodes
        metrics.completed_nodes = completed_nodes
        metrics.failed_nodes = failed_nodes
        
        if total_nodes > 0:
            metrics.success_rate = completed_nodes / total_nodes
        
        # Calculate average execution time
        completed_times = []
        for node in nodes.values():
            if node.start_time and node.end_time:
                execution_time = (node.end_time - node.start_time).total_seconds()
                completed_times.append(execution_time)
        
        if completed_times:
            metrics.average_execution_time = sum(completed_times) / len(completed_times)
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        def monitor():
            while True:
                try:
                    self._perform_health_checks()
                    self._check_for_adaptations()
                    time.sleep(self.config['health_check_interval'])
                except Exception as e:
                    print(f"Monitoring error: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def _perform_health_checks(self):
        """Perform health checks on active workflows."""
        for workflow_id, workflow in self.workflows.items():
            if workflow['state'] == WorkflowState.EXECUTING:
                # Check for timeouts
                self._check_node_timeouts(workflow_id)
                
                # Check resource utilization
                self._check_resource_utilization(workflow_id)
                
                # Check for bottlenecks
                self._identify_workflow_bottlenecks(workflow_id)
    
    def _check_node_timeouts(self, workflow_id: str):
        """Check for node execution timeouts."""
        workflow = self.workflows[workflow_id]
        current_time = datetime.now()
        
        for node in workflow['nodes'].values():
            if node.state == NodeState.RUNNING and node.start_time:
                execution_time = (current_time - node.start_time).total_seconds()
                if execution_time > self.config['timeout_threshold']:
                    # Timeout detected
                    self._handle_node_timeout(workflow_id, node.agent_name)
    
    def _handle_node_timeout(self, workflow_id: str, agent_name: str):
        """Handle node execution timeout."""
        workflow = self.workflows[workflow_id]
        node = workflow['nodes'][agent_name]
        
        # Mark as failed due to timeout
        node.state = NodeState.FAILED
        node.end_time = datetime.now()
        node.error_message = "Execution timeout"
        
        self._log_error(workflow_id, f"Node {agent_name} timed out")
        
        # Trigger recovery
        self._handle_node_failure(workflow_id, agent_name)
    
    def _check_resource_utilization(self, workflow_id: str):
        """Check resource utilization for workflow."""
        # Placeholder for resource monitoring
        # In real implementation, would monitor CPU, memory, etc.
        pass
    
    def _identify_workflow_bottlenecks(self, workflow_id: str):
        """Identify bottlenecks in workflow execution."""
        workflow = self.workflows[workflow_id]
        bottlenecks = []
        
        # Check for nodes with many dependencies waiting
        for agent_name, node in workflow['nodes'].items():
            if node.state == NodeState.PENDING:
                waiting_dependencies = sum(1 for dep in node.dependencies 
                                         if workflow['nodes'][dep].state != NodeState.COMPLETED)
                if waiting_dependencies > 1:
                    bottlenecks.append(f"Node {agent_name} waiting for {waiting_dependencies} dependencies")
        
        workflow['metrics'].bottlenecks = bottlenecks
    
    def _check_for_adaptations(self):
        """Check if workflow adaptations are needed."""
        for workflow_id, workflow in self.workflows.items():
            if workflow['state'] == WorkflowState.EXECUTING:
                # Check if adaptation is needed
                if self._adaptation_needed(workflow_id):
                    self._adapt_workflow(workflow_id)
    
    def _adaptation_needed(self, workflow_id: str) -> bool:
        """Determine if workflow adaptation is needed."""
        workflow = self.workflows[workflow_id]
        metrics = workflow['metrics']
        
        # Check failure rate
        if metrics.total_nodes > 0:
            failure_rate = metrics.failed_nodes / metrics.total_nodes
            if failure_rate > self.config['adaptation_threshold']:
                return True
        
        # Check for persistent bottlenecks
        if len(metrics.bottlenecks) > 2:
            return True
        
        return False
    
    def _adapt_workflow(self, workflow_id: str):
        """Adapt workflow based on current conditions."""
        workflow = self.workflows[workflow_id]
        workflow['state'] = WorkflowState.ADAPTING
        
        try:
            # Re-evaluate team composition
            goal_spec = workflow['goal_specification']
            preferences = workflow['preferences']
            
            # Get current performance data
            current_metrics = self._get_workflow_metrics(workflow_id)
            
            # Update preferences based on current performance
            adapted_preferences = preferences.copy()
            adapted_preferences['reliability_priority'] = 0.6  # Increase reliability focus
            adapted_preferences['speed_priority'] = 0.2  # Reduce speed focus
            
            # Compose new team
            team_input = {
                'objective': goal_spec,
                'required_capabilities': goal_spec.get('required_capabilities', []),
                'constraints': adapted_preferences,
                'preferences': adapted_preferences
            }
            
            new_team_result = self.team_composer._process(TeamComposerInput(**team_input))
            new_team_composition = new_team_result.team_composition
            
            # Update workflow with new team
            self._update_workflow_team(workflow_id, new_team_composition)
            
            workflow['state'] = WorkflowState.EXECUTING
            self._log_execution(workflow_id, "workflow", "adapted_team_composition")
            
        except Exception as e:
            self._log_error(workflow_id, f"Adaptation failed: {str(e)}")
            workflow['state'] = WorkflowState.EXECUTING  # Resume normal execution
    
    def _update_workflow_team(self, workflow_id: str, new_team_composition: Dict[str, Any]):
        """Update workflow with new team composition."""
        # Placeholder for team update logic
        # In real implementation, would update nodes and graph structure
        pass
    
    def _get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get current workflow metrics."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {}
        
        metrics = workflow['metrics']
        
        return {
            'start_time': metrics.start_time.isoformat() if metrics.start_time else None,
            'end_time': metrics.end_time.isoformat() if metrics.end_time else None,
            'total_nodes': metrics.total_nodes,
            'completed_nodes': metrics.completed_nodes,
            'failed_nodes': metrics.failed_nodes,
            'success_rate': metrics.success_rate,
            'average_execution_time': metrics.average_execution_time,
            'bottlenecks': metrics.bottlenecks,
            'progress_percentage': (metrics.completed_nodes / metrics.total_nodes * 100) if metrics.total_nodes > 0 else 0,
            'state': workflow['state'].value
        }
    
    def _log_execution(self, workflow_id: str, agent_name: str, event: str, data: Optional[Dict[str, Any]] = None):
        """Log workflow execution event."""
        workflow = self.workflows.get(workflow_id)
        if workflow:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'workflow_id': workflow_id,
                'agent_name': agent_name,
                'event': event,
                'data': data or {}
            }
            workflow['execution_log'].append(log_entry)
    
    def _log_error(self, workflow_id: str, error_message: str):
        """Log workflow error."""
        self._log_execution(workflow_id, "system", "error", {'message': error_message})
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {'error': 'Workflow not found'}
        
        return {
            'workflow_id': workflow_id,
            'state': workflow['state'].value,
            'metrics': self._get_workflow_metrics(workflow_id),
            'nodes': {name: {
                'state': node.state.value,
                'start_time': node.start_time.isoformat() if node.start_time else None,
                'end_time': node.end_time.isoformat() if node.end_time else None,
                'error_message': node.error_message,
                'retry_count': node.retry_count
            } for name, node in workflow['nodes'].items()},
            'execution_log': workflow['execution_log'][-10:]  # Last 10 events
        }
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause workflow execution."""
        workflow = self.workflows.get(workflow_id)
        if workflow and workflow['state'] == WorkflowState.EXECUTING:
            workflow['state'] = WorkflowState.PAUSED
            self._log_execution(workflow_id, "system", "paused")
            return True
        return False
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume paused workflow."""
        workflow = self.workflows.get(workflow_id)
        if workflow and workflow['state'] == WorkflowState.PAUSED:
            workflow['state'] = WorkflowState.EXECUTING
            self._log_execution(workflow_id, "system", "resumed")
            return True
        return False
    
    def stop_workflow(self, workflow_id: str) -> bool:
        """Stop workflow execution."""
        workflow = self.workflows.get(workflow_id)
        if workflow:
            workflow['state'] = WorkflowState.FAILED
            workflow['metrics'].end_time = datetime.now()
            self._log_execution(workflow_id, "system", "stopped")
            
            # Cleanup active thread
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            return True
        return False
