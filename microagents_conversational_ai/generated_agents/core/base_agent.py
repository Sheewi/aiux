"""
Base MicroAgent Framework
Production-grade microagent architecture with strict typing, error handling, and observability.
Generated on: 2025-08-13T01:48:42.506499
"""

import time
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import logging

class ErrorHandlingPolicy(str, Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    FAIL_FAST = "fail_fast"
    NOTIFY = "notify"

class MicroAgentError(Exception):
    def __init__(self, agent_name: str, original_error: Exception, context: Dict):
        self.agent_name = agent_name
        self.original_error = original_error
        self.context = context
        super().__init__(f"Agent {agent_name} failed: {str(original_error)}")

class AgentState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class BaseInput(BaseModel):
    """Base input model for all microagents."""
    agent_id: Optional[str] = Field(default=None, description="Unique identifier for this agent instance")
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout in seconds")
    priority: int = Field(default=5, ge=1, le=10, description="Priority level (1=lowest, 10=highest)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class BaseOutput(BaseModel):
    """Base output model for all microagents."""
    status: str
    agent_name: str
    execution_time: float
    timestamp: str
    result: Any
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MicroAgent(ABC):
    """Production-grade base microagent with comprehensive error handling and observability."""
    
    def __init__(self, name: str, description: str, config: Optional[Dict] = None):
        self.name = name
        self.description = description
        self.config = config or {}
        self.state = AgentState.IDLE
        self.error_policy = ErrorHandlingPolicy.RETRY
        self.max_retries = 3
        self.fallback_agent = None
        self.logger = self._setup_logger()
        
        # Metrics
        self.metrics = {
            "execution_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
        
    def _setup_logger(self):
        """Setup structured logging for the agent."""
        logger = logging.getLogger(f"microagent.{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def execute(self, input_data: Union[Dict, BaseInput]) -> BaseOutput:
        """Execute the microagent with comprehensive error handling and metrics."""
        start_time = time.time()
        attempt = 0
        last_error = None
        
        # Convert input to dict if it's a Pydantic model
        if isinstance(input_data, BaseModel):
            input_data = input_data.dict()
        
        # Validate input using the agent's input model if defined
        if hasattr(self, 'input_model'):
            try:
                validated_input = self.input_model(**input_data)
                input_data = validated_input.dict()
            except Exception as e:
                return self._create_error_output(start_time, f"Input validation failed: {str(e)}")
        
        while attempt <= self.max_retries:
            try:
                self.state = AgentState.RUNNING
                self.logger.info(f"Executing {self.name} (attempt {attempt + 1}/{self.max_retries + 1})")
                
                # Execute the agent's main logic
                result = self._process(input_data)
                
                # Update metrics and state
                execution_time = time.time() - start_time
                self._update_metrics(execution_time, success=True)
                self.state = AgentState.COMPLETED
                
                # Create successful output
                output = BaseOutput(
                    status="success",
                    agent_name=self.name,
                    execution_time=execution_time,
                    timestamp=datetime.utcnow().isoformat(),
                    result=result,
                    metadata={
                        "attempt": attempt + 1,
                        "config": self.config,
                        "metrics": self.metrics
                    }
                )
                
                self.logger.info(f"{self.name} completed successfully in {execution_time:.2f}s")
                return output
                
            except Exception as e:
                attempt += 1
                last_error = MicroAgentError(self.name, e, {"attempt": attempt, "input": input_data})
                self.logger.error(f"{self.name} failed on attempt {attempt}: {str(e)}")
                
                if self.error_policy == ErrorHandlingPolicy.FAIL_FAST or attempt > self.max_retries:
                    break
                    
                # Exponential backoff for retries
                if attempt <= self.max_retries:
                    backoff_time = min(2 ** attempt, 10)
                    self.logger.info(f"Retrying {self.name} in {backoff_time} seconds...")
                    time.sleep(backoff_time)
        
        # Handle final failure
        execution_time = time.time() - start_time
        self._update_metrics(execution_time, success=False)
        self.state = AgentState.FAILED
        
        # Try fallback agent if configured
        if self.fallback_agent and hasattr(self, '_execute_fallback'):
            try:
                return self._execute_fallback(input_data)
            except Exception as fallback_error:
                self.logger.error(f"Fallback agent also failed: {str(fallback_error)}")
        
        return self._create_error_output(start_time, str(last_error))
    
    def _update_metrics(self, execution_time: float, success: bool):
        """Update agent execution metrics."""
        self.metrics["execution_count"] += 1
        self.metrics["total_execution_time"] += execution_time
        self.metrics["average_execution_time"] = (
            self.metrics["total_execution_time"] / self.metrics["execution_count"]
        )
        
        if success:
            self.metrics["success_count"] += 1
        else:
            self.metrics["failure_count"] += 1
    
    def _create_error_output(self, start_time: float, error_message: str) -> BaseOutput:
        """Create a standardized error output."""
        execution_time = time.time() - start_time
        return BaseOutput(
            status="failed",
            agent_name=self.name,
            execution_time=execution_time,
            timestamp=datetime.utcnow().isoformat(),
            result=None,
            errors=[error_message],
            metadata={
                "config": self.config,
                "metrics": self.metrics
            }
        )
    
    @abstractmethod
    def _process(self, input_data: Dict) -> Any:
        """
        Main processing logic for the microagent.
        Must be implemented by each specific agent.
        
        Args:
            input_data: Validated input data dictionary
            
        Returns:
            Processing result (can be any type)
            
        Raises:
            Exception: If processing fails
        """
        pass
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get current health status of the agent."""
        total_executions = self.metrics["execution_count"]
        success_rate = (
            self.metrics["success_count"] / total_executions 
            if total_executions > 0 else 1.0
        )
        
        health_status = "healthy"
        if success_rate < 0.5:
            health_status = "critical"
        elif success_rate < 0.8:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "state": self.state.value,
            "success_rate": success_rate,
            "metrics": self.metrics,
            "last_error": getattr(self, 'last_error', None)
        }
    
    def reset_metrics(self):
        """Reset all metrics to initial state."""
        self.metrics = {
            "execution_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
        self.logger.info(f"Metrics reset for {self.name}")

class HybridAgent(MicroAgent):
    """Base class for hybrid agents that combine multiple microagents."""
    
    def __init__(self, name: str, description: str, component_agents: List[str], config: Optional[Dict] = None):
        super().__init__(name, description, config)
        self.component_agents = component_agents
        self.orchestration_strategy = config.get('orchestration_strategy', 'sequential') if config else 'sequential'
    
    def _execute_components(self, input_data: Dict) -> Dict[str, Any]:
        """Execute component agents based on orchestration strategy."""
        results = {}
        
        if self.orchestration_strategy == 'sequential':
            current_input = input_data
            for agent_name in self.component_agents:
                # This would need to be implemented with proper agent registry
                # For now, this is a placeholder
                result = self._execute_single_component(agent_name, current_input)
                results[agent_name] = result
                # Pass result as input to next agent
                current_input = result if isinstance(result, dict) else {"data": result}
                
        elif self.orchestration_strategy == 'parallel':
            # Execute all components in parallel with same input
            for agent_name in self.component_agents:
                result = self._execute_single_component(agent_name, input_data)
                results[agent_name] = result
                
        return results
    
    def _execute_single_component(self, agent_name: str, input_data: Dict) -> Any:
        """Execute a single component agent. Override in specific implementations."""
        # This is a placeholder - actual implementation would use agent registry
        self.logger.warning(f"Component execution not implemented for {agent_name}")
        return {"status": "not_implemented", "agent": agent_name}
