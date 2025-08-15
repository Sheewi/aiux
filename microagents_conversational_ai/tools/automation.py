import asyncio
import logging
import json
import subprocess
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import os
import signal

# Try to import schedule, fallback to basic implementation if not available
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    # Basic schedule implementation fallback
    class MockSchedule:
        def every(self, interval=1):
            return self
        
        def seconds(self):
            return self
        
        def minutes(self):
            return self
        
        def hours(self):
            return self
        
        def day(self):
            return self
        
        def monday(self):
            return self
        
        def tuesday(self):
            return self
        
        def wednesday(self):
            return self
        
        def thursday(self):
            return self
        
        def friday(self):
            return self
        
        def saturday(self):
            return self
        
        def sunday(self):
            return self
        
        def at(self, time):
            return self
        
        def do(self, func, *args):
            return None
        
        def run_pending(self):
            pass
        
        def clear(self, tag=None):
            pass
    
    schedule = MockSchedule()

from .base_tool import BaseTool, ToolStatus, ToolMetadata, ToolType, ToolCapability, create_tool_metadata


class AutomationTool(BaseTool):
    """Comprehensive automation tool for scheduling, workflow management, and task automation."""
    
    def __init__(self, max_concurrent_tasks: int = 10, config: Dict[str, Any] = None):
        """
        Initialize the automation tool.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks
            config: Additional configuration
        """
        # Initialize metadata
        metadata = create_tool_metadata(
            tool_id="automation",
            name="Automation Tool",
            description="Comprehensive automation tool for scheduling tasks, workflow management, and process automation",
            tool_type=ToolType.AUTOMATION,
            version="1.0.0",
            author="System",
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.STATEFUL,
                ToolCapability.BATCH_PROCESSING
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["schedule_task", "cancel_task", "list_tasks", "execute_workflow", "create_workflow"]},
                    "task_id": {"type": "string", "description": "Unique task identifier"},
                    "task_type": {"type": "string", "enum": ["command", "function", "workflow"]},
                    "schedule_type": {"type": "string", "enum": ["once", "daily", "weekly", "monthly", "interval"]},
                    "schedule_config": {"type": "object", "description": "Schedule configuration"},
                    "task_config": {"type": "object", "description": "Task configuration"}
                },
                "required": ["operation"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "result": {"type": "object"},
                    "error": {"type": "string"}
                }
            },
            timeout=300.0,
            supported_formats=["json", "yaml"],
            tags=["automation", "scheduling", "workflow", "tasks", "cron"]
        )
        
        super().__init__(metadata, config)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.logger = logging.getLogger(__name__)
        self.scheduled_jobs = {}
        self.running_tasks = {}
        self.task_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.scheduler_thread = None
        self.scheduler_running = False
    
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute automation operation.
        
        Args:
            operation: Type of automation operation
            **kwargs: Operation-specific parameters
        """
        self.status = ToolStatus.RUNNING
        
        try:
            operation_map = {
                'schedule_task': self._schedule_task,
                'cancel_task': self._cancel_task,
                'list_tasks': self._list_tasks,
                'run_workflow': self._run_workflow,
                'execute_command': self._execute_command,
                'monitor_process': self._monitor_process,
                'batch_process': self._batch_process,
                'conditional_execution': self._conditional_execution,
                'retry_with_backoff': self._retry_with_backoff,
                'parallel_execution': self._parallel_execution,
                'start_scheduler': self._start_scheduler,
                'stop_scheduler': self._stop_scheduler
            }
            
            if operation not in operation_map:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = await operation_map[operation](**kwargs)
            self.status = ToolStatus.COMPLETED
            return {
                'success': True,
                'operation': operation,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.status = ToolStatus.FAILED
            self.logger.error(f"Automation operation failed: {e}")
            return {
                'success': False,
                'operation': operation,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _schedule_task(self, task_id: str,
                            task_type: str,
                            schedule_type: str,
                            schedule_config: Dict[str, Any],
                            task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a task for execution."""
        
        # Validate schedule configuration
        if schedule_type not in ['once', 'interval', 'cron', 'daily', 'weekly', 'monthly']:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")
        
        # Create task definition
        task_definition = {
            'task_id': task_id,
            'task_type': task_type,
            'schedule_type': schedule_type,
            'schedule_config': schedule_config,
            'task_config': task_config,
            'created_at': datetime.now().isoformat(),
            'status': 'scheduled',
            'execution_count': 0,
            'last_execution': None,
            'next_execution': None
        }
        
        # Set up schedule based on type
        if schedule_type == 'once':
            # Schedule for one-time execution
            execute_at = datetime.fromisoformat(schedule_config['execute_at'])
            task_definition['next_execution'] = execute_at.isoformat()
            
            # Use threading timer for one-time execution
            delay = (execute_at - datetime.now()).total_seconds()
            if delay > 0:
                timer = threading.Timer(delay, self._execute_scheduled_task, [task_id])
                timer.start()
                task_definition['timer'] = timer
        
        elif schedule_type == 'interval':
            # Schedule for repeated execution
            interval_seconds = schedule_config.get('seconds', 0)
            interval_minutes = schedule_config.get('minutes', 0)
            interval_hours = schedule_config.get('hours', 0)
            
            total_seconds = interval_seconds + (interval_minutes * 60) + (interval_hours * 3600)
            
            if total_seconds <= 0:
                raise ValueError("Interval must be greater than 0")
            
            # Calculate next execution
            next_exec = datetime.now() + timedelta(seconds=total_seconds)
            task_definition['next_execution'] = next_exec.isoformat()
            
            # Set up recurring schedule using schedule library
            if interval_seconds > 0 and interval_minutes == 0 and interval_hours == 0:
                schedule.every(interval_seconds).seconds.do(self._execute_scheduled_task, task_id)
            elif interval_minutes > 0 and interval_hours == 0:
                schedule.every(interval_minutes).minutes.do(self._execute_scheduled_task, task_id)
            elif interval_hours > 0:
                schedule.every(interval_hours).hours.do(self._execute_scheduled_task, task_id)
        
        elif schedule_type == 'daily':
            # Daily execution at specific time
            execute_time = schedule_config.get('time', '00:00')
            schedule.every().day.at(execute_time).do(self._execute_scheduled_task, task_id)
            
            # Calculate next execution
            today = datetime.now().date()
            exec_time = datetime.strptime(execute_time, '%H:%M').time()
            next_exec = datetime.combine(today, exec_time)
            if next_exec <= datetime.now():
                next_exec += timedelta(days=1)
            task_definition['next_execution'] = next_exec.isoformat()
        
        elif schedule_type == 'weekly':
            # Weekly execution on specific day and time
            day_of_week = schedule_config.get('day', 'monday').lower()
            execute_time = schedule_config.get('time', '00:00')
            
            # Map day names to schedule methods
            day_methods = {
                'monday': schedule.every().monday,
                'tuesday': schedule.every().tuesday,
                'wednesday': schedule.every().wednesday,
                'thursday': schedule.every().thursday,
                'friday': schedule.every().friday,
                'saturday': schedule.every().saturday,
                'sunday': schedule.every().sunday
            }
            
            if day_of_week not in day_methods:
                raise ValueError(f"Invalid day of week: {day_of_week}")
            
            day_methods[day_of_week].at(execute_time).do(self._execute_scheduled_task, task_id)
        
        # Store task definition
        self.scheduled_jobs[task_id] = task_definition
        
        # Start scheduler if not running
        if not self.scheduler_running:
            await self._start_scheduler()
        
        return {
            'task_id': task_id,
            'status': 'scheduled',
            'schedule_type': schedule_type,
            'next_execution': task_definition.get('next_execution'),
            'created_at': task_definition['created_at']
        }
    
    def _execute_scheduled_task(self, task_id: str):
        """Execute a scheduled task."""
        if task_id not in self.scheduled_jobs:
            self.logger.error(f"Task {task_id} not found")
            return
        
        task_def = self.scheduled_jobs[task_id]
        
        try:
            # Update execution info
            task_def['status'] = 'running'
            task_def['execution_count'] += 1
            task_def['last_execution'] = datetime.now().isoformat()
            
            # Execute based on task type
            task_type = task_def['task_type']
            task_config = task_def['task_config']
            
            if task_type == 'command':
                result = self._execute_command_sync(task_config.get('command', ''))
            elif task_type == 'function':
                func = task_config.get('function')
                args = task_config.get('args', [])
                kwargs = task_config.get('kwargs', {})
                result = func(*args, **kwargs) if callable(func) else None
            elif task_type == 'http_request':
                result = self._make_http_request_sync(task_config)
            else:
                result = {'error': f'Unknown task type: {task_type}'}
            
            task_def['status'] = 'completed'
            task_def['last_result'] = result
            
            self.logger.info(f"Task {task_id} executed successfully")
            
        except Exception as e:
            task_def['status'] = 'failed'
            task_def['last_error'] = str(e)
            self.logger.error(f"Task {task_id} failed: {e}")
    
    def _execute_command_sync(self, command: str) -> Dict[str, Any]:
        """Execute a system command synchronously."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                'command': command,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            return {
                'command': command,
                'error': 'Command timed out',
                'execution_time': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'command': command,
                'error': str(e),
                'execution_time': datetime.now().isoformat()
            }
    
    def _make_http_request_sync(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request synchronously."""
        import requests
        
        try:
            method = config.get('method', 'GET').upper()
            url = config.get('url', '')
            headers = config.get('headers', {})
            data = config.get('data')
            timeout = config.get('timeout', 30)
            
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data if isinstance(data, dict) else None,
                data=data if isinstance(data, str) else None,
                timeout=timeout
            )
            
            return {
                'url': url,
                'method': method,
                'status_code': response.status_code,
                'response_headers': dict(response.headers),
                'response_text': response.text[:1000],  # Limit response size
                'execution_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'url': config.get('url', ''),
                'method': config.get('method', 'GET'),
                'error': str(e),
                'execution_time': datetime.now().isoformat()
            }
    
    async def _cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a scheduled task."""
        if task_id not in self.scheduled_jobs:
            raise ValueError(f"Task {task_id} not found")
        
        task_def = self.scheduled_jobs[task_id]
        
        # Cancel timer if it exists
        if 'timer' in task_def:
            task_def['timer'].cancel()
        
        # Remove from schedule library
        schedule.clear(task_id)
        
        # Update status
        task_def['status'] = 'cancelled'
        task_def['cancelled_at'] = datetime.now().isoformat()
        
        return {
            'task_id': task_id,
            'status': 'cancelled',
            'execution_count': task_def['execution_count'],
            'cancelled_at': task_def['cancelled_at']
        }
    
    async def _list_tasks(self, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """List all scheduled tasks."""
        tasks = []
        
        for task_id, task_def in self.scheduled_jobs.items():
            if status_filter is None or task_def['status'] == status_filter:
                task_info = {
                    'task_id': task_id,
                    'task_type': task_def['task_type'],
                    'schedule_type': task_def['schedule_type'],
                    'status': task_def['status'],
                    'execution_count': task_def['execution_count'],
                    'created_at': task_def['created_at'],
                    'last_execution': task_def.get('last_execution'),
                    'next_execution': task_def.get('next_execution')
                }
                tasks.append(task_info)
        
        return {
            'total_tasks': len(tasks),
            'status_filter': status_filter,
            'tasks': tasks
        }
    
    async def _run_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Run a workflow with multiple steps."""
        workflow_id = workflow_definition.get('id', f"workflow_{int(time.time())}")
        steps = workflow_definition.get('steps', [])
        
        workflow_result = {
            'workflow_id': workflow_id,
            'total_steps': len(steps),
            'completed_steps': 0,
            'failed_steps': 0,
            'step_results': [],
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        
        try:
            for i, step in enumerate(steps):
                step_id = step.get('id', f"step_{i}")
                step_type = step.get('type', 'command')
                step_config = step.get('config', {})
                
                self.logger.info(f"Executing workflow step {step_id}")
                
                # Execute step based on type
                if step_type == 'command':
                    step_result = await self._execute_command(
                        command=step_config.get('command', ''),
                        timeout=step_config.get('timeout', 300)
                    )
                elif step_type == 'delay':
                    delay_seconds = step_config.get('seconds', 1)
                    await asyncio.sleep(delay_seconds)
                    step_result = {'success': True, 'delay_seconds': delay_seconds}
                elif step_type == 'condition':
                    condition = step_config.get('condition', True)
                    if not condition:
                        step_result = {'success': False, 'reason': 'Condition not met'}
                        if step_config.get('fail_on_false', True):
                            workflow_result['failed_steps'] += 1
                            break
                    else:
                        step_result = {'success': True, 'condition_met': True}
                else:
                    step_result = {'success': False, 'error': f'Unknown step type: {step_type}'}
                
                # Record step result
                workflow_result['step_results'].append({
                    'step_id': step_id,
                    'step_type': step_type,
                    'result': step_result,
                    'execution_time': datetime.now().isoformat()
                })
                
                # Check if step failed
                if not step_result.get('success', False):
                    workflow_result['failed_steps'] += 1
                    if step.get('fail_fast', False):
                        break
                else:
                    workflow_result['completed_steps'] += 1
            
            # Set final status
            if workflow_result['failed_steps'] > 0:
                workflow_result['status'] = 'failed'
            else:
                workflow_result['status'] = 'completed'
            
            workflow_result['end_time'] = datetime.now().isoformat()
            
            return workflow_result
            
        except Exception as e:
            workflow_result['status'] = 'error'
            workflow_result['error'] = str(e)
            workflow_result['end_time'] = datetime.now().isoformat()
            return workflow_result
    
    async def _execute_command(self, command: str, 
                              timeout: int = 300,
                              working_directory: Optional[str] = None) -> Dict[str, Any]:
        """Execute a system command asynchronously."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_directory
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                
                return {
                    'success': process.returncode == 0,
                    'command': command,
                    'return_code': process.returncode,
                    'stdout': stdout.decode('utf-8'),
                    'stderr': stderr.decode('utf-8'),
                    'execution_time': datetime.now().isoformat()
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    'success': False,
                    'command': command,
                    'error': 'Command timed out',
                    'timeout': timeout,
                    'execution_time': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'command': command,
                'error': str(e),
                'execution_time': datetime.now().isoformat()
            }
    
    async def _monitor_process(self, process_name: str,
                              monitor_duration: int = 60) -> Dict[str, Any]:
        """Monitor a system process."""
        import psutil
        
        monitoring_results = {
            'process_name': process_name,
            'monitor_duration': monitor_duration,
            'samples': [],
            'start_time': datetime.now().isoformat()
        }
        
        start_time = time.time()
        
        while time.time() - start_time < monitor_duration:
            try:
                # Find processes by name
                processes = [p for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info'])
                           if process_name.lower() in p.info['name'].lower()]
                
                if processes:
                    sample = {
                        'timestamp': datetime.now().isoformat(),
                        'processes': []
                    }
                    
                    for proc in processes:
                        sample['processes'].append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024
                        })
                    
                    monitoring_results['samples'].append(sample)
                else:
                    monitoring_results['samples'].append({
                        'timestamp': datetime.now().isoformat(),
                        'processes': [],
                        'note': 'No processes found'
                    })
                
                await asyncio.sleep(5)  # Sample every 5 seconds
                
            except Exception as e:
                monitoring_results['samples'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                })
        
        monitoring_results['end_time'] = datetime.now().isoformat()
        monitoring_results['total_samples'] = len(monitoring_results['samples'])
        
        return monitoring_results
    
    async def _batch_process(self, items: List[Any],
                            operation: str,
                            operation_config: Dict[str, Any],
                            batch_size: int = 10,
                            parallel: bool = True) -> Dict[str, Any]:
        """Process items in batches."""
        total_items = len(items)
        processed_items = 0
        failed_items = 0
        results = []
        
        batch_result = {
            'total_items': total_items,
            'batch_size': batch_size,
            'parallel_processing': parallel,
            'start_time': datetime.now().isoformat(),
            'results': []
        }
        
        # Process items in batches
        for i in range(0, total_items, batch_size):
            batch = items[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            self.logger.info(f"Processing batch {batch_num}, items {i+1}-{min(i+batch_size, total_items)}")
            
            if parallel:
                # Process batch items in parallel
                tasks = []
                for item in batch:
                    task = self._process_single_item(item, operation, operation_config)
                    tasks.append(task)
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Process batch items sequentially
                batch_results = []
                for item in batch:
                    result = await self._process_single_item(item, operation, operation_config)
                    batch_results.append(result)
            
            # Collect results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    failed_items += 1
                    results.append({
                        'item_index': i + j,
                        'success': False,
                        'error': str(result)
                    })
                else:
                    processed_items += 1
                    results.append({
                        'item_index': i + j,
                        'success': True,
                        'result': result
                    })
        
        batch_result.update({
            'processed_items': processed_items,
            'failed_items': failed_items,
            'success_rate': processed_items / total_items if total_items > 0 else 0,
            'results': results,
            'end_time': datetime.now().isoformat()
        })
        
        return batch_result
    
    async def _process_single_item(self, item: Any, 
                                  operation: str,
                                  operation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item."""
        try:
            if operation == 'transform':
                transform_func = operation_config.get('function')
                if callable(transform_func):
                    result = transform_func(item)
                else:
                    result = item
            elif operation == 'validate':
                validation_func = operation_config.get('function')
                if callable(validation_func):
                    result = {'valid': validation_func(item), 'item': item}
                else:
                    result = {'valid': True, 'item': item}
            elif operation == 'save':
                # Simulate saving operation
                await asyncio.sleep(0.1)
                result = {'saved': True, 'item': item}
            else:
                result = {'processed': True, 'item': item}
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to process item {item}: {e}")
    
    async def _conditional_execution(self, condition: Union[bool, Callable],
                                    if_true_action: Dict[str, Any],
                                    if_false_action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute actions based on condition."""
        # Evaluate condition
        if callable(condition):
            condition_result = condition()
        else:
            condition_result = bool(condition)
        
        execution_result = {
            'condition_result': condition_result,
            'executed_action': None,
            'action_result': None,
            'execution_time': datetime.now().isoformat()
        }
        
        # Execute appropriate action
        if condition_result and if_true_action:
            execution_result['executed_action'] = 'if_true'
            execution_result['action_result'] = await self._execute_action(if_true_action)
        elif not condition_result and if_false_action:
            execution_result['executed_action'] = 'if_false'
            execution_result['action_result'] = await self._execute_action(if_false_action)
        else:
            execution_result['executed_action'] = 'none'
            execution_result['action_result'] = {'message': 'No action executed'}
        
        return execution_result
    
    async def _execute_action(self, action_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action."""
        action_type = action_config.get('type', 'command')
        
        if action_type == 'command':
            return await self._execute_command(action_config.get('command', ''))
        elif action_type == 'function':
            func = action_config.get('function')
            args = action_config.get('args', [])
            kwargs = action_config.get('kwargs', {})
            
            if callable(func):
                result = func(*args, **kwargs)
                return {'success': True, 'result': result}
            else:
                return {'success': False, 'error': 'Function not callable'}
        else:
            return {'success': False, 'error': f'Unknown action type: {action_type}'}
    
    async def _retry_with_backoff(self, operation_config: Dict[str, Any],
                                 max_retries: int = 3,
                                 backoff_factor: float = 2.0,
                                 initial_delay: float = 1.0) -> Dict[str, Any]:
        """Execute operation with retry and exponential backoff."""
        retry_result = {
            'max_retries': max_retries,
            'backoff_factor': backoff_factor,
            'initial_delay': initial_delay,
            'attempts': [],
            'final_success': False,
            'start_time': datetime.now().isoformat()
        }
        
        for attempt in range(max_retries + 1):
            attempt_info = {
                'attempt_number': attempt + 1,
                'start_time': datetime.now().isoformat()
            }
            
            try:
                # Execute the operation
                result = await self._execute_action(operation_config)
                
                attempt_info['result'] = result
                attempt_info['success'] = result.get('success', False)
                attempt_info['end_time'] = datetime.now().isoformat()
                
                retry_result['attempts'].append(attempt_info)
                
                if attempt_info['success']:
                    retry_result['final_success'] = True
                    retry_result['final_result'] = result
                    break
                else:
                    # If not the last attempt, wait before retrying
                    if attempt < max_retries:
                        delay = initial_delay * (backoff_factor ** attempt)
                        attempt_info['delay_before_next'] = delay
                        await asyncio.sleep(delay)
                
            except Exception as e:
                attempt_info['error'] = str(e)
                attempt_info['success'] = False
                attempt_info['end_time'] = datetime.now().isoformat()
                
                retry_result['attempts'].append(attempt_info)
                
                if attempt < max_retries:
                    delay = initial_delay * (backoff_factor ** attempt)
                    attempt_info['delay_before_next'] = delay
                    await asyncio.sleep(delay)
        
        retry_result['end_time'] = datetime.now().isoformat()
        retry_result['total_attempts'] = len(retry_result['attempts'])
        
        return retry_result
    
    async def _parallel_execution(self, tasks: List[Dict[str, Any]],
                                 max_concurrent: Optional[int] = None) -> Dict[str, Any]:
        """Execute multiple tasks in parallel."""
        if max_concurrent is None:
            max_concurrent = self.max_concurrent_tasks
        
        parallel_result = {
            'total_tasks': len(tasks),
            'max_concurrent': max_concurrent,
            'start_time': datetime.now().isoformat(),
            'task_results': []
        }
        
        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_task_with_semaphore(task_config: Dict[str, Any], task_index: int):
            async with semaphore:
                task_start_time = datetime.now().isoformat()
                try:
                    result = await self._execute_action(task_config)
                    return {
                        'task_index': task_index,
                        'task_id': task_config.get('id', f'task_{task_index}'),
                        'success': True,
                        'result': result,
                        'start_time': task_start_time,
                        'end_time': datetime.now().isoformat()
                    }
                except Exception as e:
                    return {
                        'task_index': task_index,
                        'task_id': task_config.get('id', f'task_{task_index}'),
                        'success': False,
                        'error': str(e),
                        'start_time': task_start_time,
                        'end_time': datetime.now().isoformat()
                    }
        
        # Execute all tasks
        task_futures = [
            execute_task_with_semaphore(task, i) 
            for i, task in enumerate(tasks)
        ]
        
        results = await asyncio.gather(*task_futures)
        
        parallel_result['task_results'] = results
        parallel_result['end_time'] = datetime.now().isoformat()
        parallel_result['successful_tasks'] = sum(1 for r in results if r['success'])
        parallel_result['failed_tasks'] = sum(1 for r in results if not r['success'])
        
        return parallel_result
    
    async def _start_scheduler(self) -> Dict[str, Any]:
        """Start the task scheduler."""
        if self.scheduler_running:
            return {'status': 'already_running'}
        
        def run_scheduler():
            self.scheduler_running = True
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(1)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        return {
            'status': 'started',
            'start_time': datetime.now().isoformat()
        }
    
    async def _stop_scheduler(self) -> Dict[str, Any]:
        """Stop the task scheduler."""
        if not self.scheduler_running:
            return {'status': 'not_running'}
        
        self.scheduler_running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        return {
            'status': 'stopped',
            'stop_time': datetime.now().isoformat()
        }
