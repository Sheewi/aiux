import asyncio
import psutil
import logging
import json
import time
import statistics
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import threading
import queue
import os
import socket
import requests
from collections import defaultdict, deque

from .base_tool import BaseTool, ToolStatus, ToolMetadata, ToolType, ToolCapability, create_tool_metadata


class MonitoringTool(BaseTool):
    """Comprehensive monitoring tool for system metrics, performance, and health checks."""
    
    def __init__(self, history_size: int = 1000, config: Dict[str, Any] = None):
        """
        Initialize the monitoring tool.
        
        Args:
            history_size: Maximum number of historical data points to keep
            config: Additional configuration
        """
        # Initialize metadata
        metadata = create_tool_metadata(
            tool_id="monitoring",
            name="Monitoring Tool",
            description="Comprehensive monitoring tool for system metrics, performance analysis, health checks, and alerting",
            tool_type=ToolType.MONITORING,
            version="1.0.0",
            author="System",
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.REAL_TIME,
                ToolCapability.STATEFUL,
                ToolCapability.HARDWARE_DEPENDENT
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["system_metrics", "health_check", "performance_monitor", "set_alert", "get_alerts"]},
                    "services": {"type": "array", "description": "Services to monitor"},
                    "alert_config": {"type": "object", "description": "Alert configuration"},
                    "duration": {"type": "integer", "description": "Monitoring duration in seconds"}
                },
                "required": ["operation"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "result": {"type": "object"},
                    "error": {"type": "string"},
                    "timestamp": {"type": "string"}
                }
            },
            timeout=120.0,
            supported_formats=["json", "metrics"],
            tags=["monitoring", "metrics", "performance", "health", "alerts", "system"]
        )
        
        super().__init__(metadata, config)
        self.history_size = history_size
        self.logger = logging.getLogger(__name__)
        self.metric_history = defaultdict(lambda: deque(maxlen=history_size))
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_rules = {}
        self.alerts_triggered = []
    
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute monitoring operation.
        
        Args:
            operation: Type of monitoring operation
            **kwargs: Operation-specific parameters
        """
        self.status = ToolStatus.RUNNING
        
        try:
            operation_map = {
                'system_metrics': self._get_system_metrics,
                'process_metrics': self._get_process_metrics,
                'network_metrics': self._get_network_metrics,
                'disk_metrics': self._get_disk_metrics,
                'memory_metrics': self._get_memory_metrics,
                'cpu_metrics': self._get_cpu_metrics,
                'health_check': self._perform_health_check,
                'service_monitor': self._monitor_service,
                'log_analysis': self._analyze_logs,
                'performance_report': self._generate_performance_report,
                'start_monitoring': self._start_continuous_monitoring,
                'stop_monitoring': self._stop_continuous_monitoring,
                'set_alert': self._set_alert_rule,
                'check_alerts': self._check_alert_conditions
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
            self.logger.error(f"Monitoring operation failed: {e}")
            return {
                'success': False,
                'operation': operation,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        timestamp = datetime.now().isoformat()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        # System info
        boot_time = datetime.fromtimestamp(psutil.boot_time()).isoformat()
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        
        metrics = {
            'timestamp': timestamp,
            'cpu': {
                'usage_percent': cpu_percent,
                'logical_cores': cpu_count_logical,
                'physical_cores': cpu_count_physical,
                'frequency_mhz': cpu_freq.current if cpu_freq else None,
                'load_average': {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                }
            },
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'usage_percent': memory.percent,
                'cached_gb': round(getattr(memory, 'cached', 0) / (1024**3), 2),
                'buffers_gb': round(getattr(memory, 'buffers', 0) / (1024**3), 2)
            },
            'swap': {
                'total_gb': round(swap.total / (1024**3), 2),
                'used_gb': round(swap.used / (1024**3), 2),
                'usage_percent': swap.percent
            },
            'disk': {
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'used_gb': round(disk_usage.used / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'usage_percent': (disk_usage.used / disk_usage.total) * 100,
                'io': {
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0,
                    'read_count': disk_io.read_count if disk_io else 0,
                    'write_count': disk_io.write_count if disk_io else 0
                } if disk_io else None
            },
            'network': {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv,
                'errors_in': network_io.errin,
                'errors_out': network_io.errout,
                'drops_in': network_io.dropin,
                'drops_out': network_io.dropout
            },
            'system': {
                'boot_time': boot_time,
                'uptime_seconds': time.time() - psutil.boot_time(),
                'process_count': len(psutil.pids())
            }
        }
        
        # Store in history
        self.metric_history['system_metrics'].append(metrics)
        
        return metrics
    
    async def _get_process_metrics(self, process_name: Optional[str] = None,
                                  pid: Optional[int] = None,
                                  top_n: int = 10) -> Dict[str, Any]:
        """Get process-specific metrics."""
        timestamp = datetime.now().isoformat()
        
        if pid:
            # Get metrics for specific PID
            try:
                proc = psutil.Process(pid)
                return await self._get_single_process_metrics(proc, timestamp)
            except psutil.NoSuchProcess:
                return {
                    'timestamp': timestamp,
                    'error': f'Process with PID {pid} not found'
                }
        
        elif process_name:
            # Get metrics for processes with specific name
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'create_time']):
                if process_name.lower() in proc.info['name'].lower():
                    proc_metrics = await self._get_single_process_metrics(proc, timestamp)
                    processes.append(proc_metrics['process'])
            
            return {
                'timestamp': timestamp,
                'process_name_filter': process_name,
                'matching_processes': processes,
                'process_count': len(processes)
            }
        
        else:
            # Get top N processes by CPU/memory usage
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage and get top N
            top_cpu = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:top_n]
            # Sort by memory usage and get top N
            top_memory = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:top_n]
            
            return {
                'timestamp': timestamp,
                'top_cpu_processes': top_cpu,
                'top_memory_processes': top_memory,
                'total_processes': len(processes)
            }
    
    async def _get_single_process_metrics(self, proc: psutil.Process, timestamp: str) -> Dict[str, Any]:
        """Get metrics for a single process."""
        try:
            with proc.oneshot():
                memory_info = proc.memory_info()
                cpu_times = proc.cpu_times()
                
                process_metrics = {
                    'pid': proc.pid,
                    'name': proc.name(),
                    'status': proc.status(),
                    'create_time': datetime.fromtimestamp(proc.create_time()).isoformat(),
                    'cpu': {
                        'percent': proc.cpu_percent(),
                        'user_time': cpu_times.user,
                        'system_time': cpu_times.system,
                        'num_threads': proc.num_threads()
                    },
                    'memory': {
                        'rss_mb': round(memory_info.rss / (1024**2), 2),
                        'vms_mb': round(memory_info.vms / (1024**2), 2),
                        'percent': proc.memory_percent(),
                        'shared_mb': round(getattr(memory_info, 'shared', 0) / (1024**2), 2)
                    },
                    'io': None,
                    'connections': []
                }
                
                # Get I/O counters if available
                try:
                    io_counters = proc.io_counters()
                    process_metrics['io'] = {
                        'read_bytes': io_counters.read_bytes,
                        'write_bytes': io_counters.write_bytes,
                        'read_count': io_counters.read_count,
                        'write_count': io_counters.write_count
                    }
                except (psutil.AccessDenied, AttributeError):
                    pass
                
                # Get network connections if available
                try:
                    connections = proc.connections()
                    process_metrics['connections'] = [
                        {
                            'fd': conn.fd,
                            'family': str(conn.family),
                            'type': str(conn.type),
                            'local_address': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                            'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                            'status': conn.status
                        }
                        for conn in connections[:10]  # Limit to 10 connections
                    ]
                except (psutil.AccessDenied, AttributeError):
                    pass
                
                return {
                    'timestamp': timestamp,
                    'process': process_metrics
                }
                
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            return {
                'timestamp': timestamp,
                'error': str(e)
            }
    
    async def _get_network_metrics(self) -> Dict[str, Any]:
        """Get detailed network metrics."""
        timestamp = datetime.now().isoformat()
        
        # Overall network I/O
        net_io = psutil.net_io_counters()
        
        # Per-interface statistics
        net_io_per_interface = psutil.net_io_counters(pernic=True)
        
        # Network connections
        connections = psutil.net_connections()
        
        # Group connections by status
        conn_by_status = defaultdict(int)
        conn_by_family = defaultdict(int)
        
        for conn in connections:
            conn_by_status[conn.status] += 1
            conn_by_family[str(conn.family)] += 1
        
        # Interface details
        interfaces = []
        net_if_addrs = psutil.net_if_addrs()
        net_if_stats = psutil.net_if_stats()
        
        for interface_name, interface_io in net_io_per_interface.items():
            interface_info = {
                'name': interface_name,
                'bytes_sent': interface_io.bytes_sent,
                'bytes_recv': interface_io.bytes_recv,
                'packets_sent': interface_io.packets_sent,
                'packets_recv': interface_io.packets_recv,
                'errors_in': interface_io.errin,
                'errors_out': interface_io.errout,
                'drops_in': interface_io.dropin,
                'drops_out': interface_io.dropout,
                'addresses': [],
                'is_up': False,
                'speed_mbps': None
            }
            
            # Add address information
            if interface_name in net_if_addrs:
                for addr in net_if_addrs[interface_name]:
                    interface_info['addresses'].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })
            
            # Add interface status and speed
            if interface_name in net_if_stats:
                stats = net_if_stats[interface_name]
                interface_info['is_up'] = stats.isup
                interface_info['speed_mbps'] = stats.speed
            
            interfaces.append(interface_info)
        
        return {
            'timestamp': timestamp,
            'overall': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errors_in': net_io.errin,
                'errors_out': net_io.errout,
                'drops_in': net_io.dropin,
                'drops_out': net_io.dropout
            },
            'interfaces': interfaces,
            'connections': {
                'total': len(connections),
                'by_status': dict(conn_by_status),
                'by_family': dict(conn_by_family)
            }
        }
    
    async def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get detailed disk metrics."""
        timestamp = datetime.now().isoformat()
        
        # Disk usage for all mount points
        disk_usage = []
        disk_partitions = psutil.disk_partitions()
        
        for partition in disk_partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'filesystem': partition.fstype,
                    'total_gb': round(usage.total / (1024**3), 2),
                    'used_gb': round(usage.used / (1024**3), 2),
                    'free_gb': round(usage.free / (1024**3), 2),
                    'usage_percent': round((usage.used / usage.total) * 100, 2)
                })
            except (PermissionError, FileNotFoundError):
                # Skip inaccessible mount points
                continue
        
        # Disk I/O statistics
        disk_io = psutil.disk_io_counters()
        disk_io_per_disk = psutil.disk_io_counters(perdisk=True)
        
        # Per-disk I/O
        disk_io_details = []
        for disk_name, io_stats in disk_io_per_disk.items():
            disk_io_details.append({
                'disk': disk_name,
                'read_bytes': io_stats.read_bytes,
                'write_bytes': io_stats.write_bytes,
                'read_count': io_stats.read_count,
                'write_count': io_stats.write_count,
                'read_time_ms': io_stats.read_time,
                'write_time_ms': io_stats.write_time
            })
        
        return {
            'timestamp': timestamp,
            'usage': disk_usage,
            'io_overall': {
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0,
                'read_time_ms': disk_io.read_time if disk_io else 0,
                'write_time_ms': disk_io.write_time if disk_io else 0
            } if disk_io else None,
            'io_per_disk': disk_io_details
        }
    
    async def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get detailed memory metrics."""
        timestamp = datetime.now().isoformat()
        
        # Virtual memory
        vmem = psutil.virtual_memory()
        
        # Swap memory
        swap = psutil.swap_memory()
        
        # Memory by process (top 10)
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'memory_percent']):
            try:
                memory_info = proc.info['memory_info']
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'rss_mb': round(memory_info.rss / (1024**2), 2),
                    'vms_mb': round(memory_info.vms / (1024**2), 2),
                    'memory_percent': proc.info['memory_percent']
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by memory usage and get top 10
        top_memory_processes = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:10]
        
        return {
            'timestamp': timestamp,
            'virtual_memory': {
                'total_gb': round(vmem.total / (1024**3), 2),
                'available_gb': round(vmem.available / (1024**3), 2),
                'used_gb': round(vmem.used / (1024**3), 2),
                'free_gb': round(vmem.free / (1024**3), 2),
                'usage_percent': vmem.percent,
                'active_gb': round(getattr(vmem, 'active', 0) / (1024**3), 2),
                'inactive_gb': round(getattr(vmem, 'inactive', 0) / (1024**3), 2),
                'cached_gb': round(getattr(vmem, 'cached', 0) / (1024**3), 2),
                'buffers_gb': round(getattr(vmem, 'buffers', 0) / (1024**3), 2),
                'shared_gb': round(getattr(vmem, 'shared', 0) / (1024**3), 2)
            },
            'swap_memory': {
                'total_gb': round(swap.total / (1024**3), 2),
                'used_gb': round(swap.used / (1024**3), 2),
                'free_gb': round(swap.free / (1024**3), 2),
                'usage_percent': swap.percent,
                'swap_in': getattr(swap, 'sin', 0),
                'swap_out': getattr(swap, 'sout', 0)
            },
            'top_memory_processes': top_memory_processes
        }
    
    async def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get detailed CPU metrics."""
        timestamp = datetime.now().isoformat()
        
        # Overall CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
        
        # CPU times
        cpu_times = psutil.cpu_times()
        cpu_times_per_core = psutil.cpu_times(percpu=True)
        
        # CPU frequency
        cpu_freq = psutil.cpu_freq()
        cpu_freq_per_core = psutil.cpu_freq(percpu=True) if hasattr(psutil, 'cpu_freq') else []
        
        # CPU statistics
        cpu_stats = psutil.cpu_stats()
        
        # Load average (Unix-like systems)
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        
        # Top CPU-consuming processes
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'cpu_times']):
            try:
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu_percent': proc.info['cpu_percent']
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        top_cpu_processes = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:10]
        
        return {
            'timestamp': timestamp,
            'overall': {
                'usage_percent': cpu_percent,
                'logical_cores': psutil.cpu_count(logical=True),
                'physical_cores': psutil.cpu_count(logical=False),
                'frequency_current_mhz': cpu_freq.current if cpu_freq else None,
                'frequency_min_mhz': cpu_freq.min if cpu_freq else None,
                'frequency_max_mhz': cpu_freq.max if cpu_freq else None
            },
            'per_core': {
                'usage_percent': cpu_percent_per_core,
                'frequencies_mhz': [freq.current for freq in cpu_freq_per_core] if cpu_freq_per_core else []
            },
            'times': {
                'user': cpu_times.user,
                'system': cpu_times.system,
                'idle': cpu_times.idle,
                'nice': getattr(cpu_times, 'nice', 0),
                'iowait': getattr(cpu_times, 'iowait', 0),
                'irq': getattr(cpu_times, 'irq', 0),
                'softirq': getattr(cpu_times, 'softirq', 0),
                'steal': getattr(cpu_times, 'steal', 0),
                'guest': getattr(cpu_times, 'guest', 0)
            },
            'stats': {
                'context_switches': cpu_stats.ctx_switches,
                'interrupts': cpu_stats.interrupts,
                'soft_interrupts': cpu_stats.soft_interrupts,
                'syscalls': getattr(cpu_stats, 'syscalls', 0)
            },
            'load_average': {
                '1min': load_avg[0],
                '5min': load_avg[1],
                '15min': load_avg[2]
            },
            'top_cpu_processes': top_cpu_processes
        }
    
    async def _perform_health_check(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform health checks on services."""
        timestamp = datetime.now().isoformat()
        health_results = []
        
        for service in services:
            service_name = service.get('name', 'Unknown')
            check_type = service.get('type', 'http')
            
            if check_type == 'http':
                result = await self._http_health_check(service)
            elif check_type == 'tcp':
                result = await self._tcp_health_check(service)
            elif check_type == 'process':
                result = await self._process_health_check(service)
            elif check_type == 'disk_space':
                result = await self._disk_space_health_check(service)
            else:
                result = {
                    'service': service_name,
                    'status': 'unknown',
                    'error': f'Unknown check type: {check_type}'
                }
            
            health_results.append(result)
        
        # Calculate overall health
        healthy_count = sum(1 for r in health_results if r.get('status') == 'healthy')
        total_count = len(health_results)
        overall_health = 'healthy' if healthy_count == total_count else 'unhealthy'
        
        return {
            'timestamp': timestamp,
            'overall_health': overall_health,
            'healthy_services': healthy_count,
            'total_services': total_count,
            'health_score': round((healthy_count / total_count) * 100, 2) if total_count > 0 else 0,
            'service_results': health_results
        }
    
    async def _http_health_check(self, service: Dict[str, Any]) -> Dict[str, Any]:
        """Perform HTTP health check."""
        service_name = service.get('name', 'Unknown')
        url = service.get('url', '')
        timeout = service.get('timeout', 10)
        expected_status = service.get('expected_status', 200)
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            response_time = round((time.time() - start_time) * 1000, 2)  # ms
            
            status = 'healthy' if response.status_code == expected_status else 'unhealthy'
            
            return {
                'service': service_name,
                'type': 'http',
                'url': url,
                'status': status,
                'response_code': response.status_code,
                'response_time_ms': response_time,
                'content_length': len(response.content)
            }
            
        except requests.exceptions.Timeout:
            return {
                'service': service_name,
                'type': 'http',
                'url': url,
                'status': 'unhealthy',
                'error': 'Request timeout'
            }
        except requests.exceptions.ConnectionError:
            return {
                'service': service_name,
                'type': 'http',
                'url': url,
                'status': 'unhealthy',
                'error': 'Connection error'
            }
        except Exception as e:
            return {
                'service': service_name,
                'type': 'http',
                'url': url,
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _tcp_health_check(self, service: Dict[str, Any]) -> Dict[str, Any]:
        """Perform TCP port health check."""
        service_name = service.get('name', 'Unknown')
        host = service.get('host', 'localhost')
        port = service.get('port', 80)
        timeout = service.get('timeout', 5)
        
        try:
            start_time = time.time()
            with socket.create_connection((host, port), timeout=timeout) as sock:
                response_time = round((time.time() - start_time) * 1000, 2)  # ms
                
                return {
                    'service': service_name,
                    'type': 'tcp',
                    'host': host,
                    'port': port,
                    'status': 'healthy',
                    'response_time_ms': response_time
                }
                
        except socket.timeout:
            return {
                'service': service_name,
                'type': 'tcp',
                'host': host,
                'port': port,
                'status': 'unhealthy',
                'error': 'Connection timeout'
            }
        except socket.error as e:
            return {
                'service': service_name,
                'type': 'tcp',
                'host': host,
                'port': port,
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _process_health_check(self, service: Dict[str, Any]) -> Dict[str, Any]:
        """Perform process health check."""
        service_name = service.get('name', 'Unknown')
        process_name = service.get('process_name', '')
        
        running_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'status', 'create_time']):
            try:
                if process_name.lower() in proc.info['name'].lower():
                    running_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'status': proc.info['status'],
                        'uptime_seconds': time.time() - proc.info['create_time']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        status = 'healthy' if running_processes else 'unhealthy'
        
        return {
            'service': service_name,
            'type': 'process',
            'process_name': process_name,
            'status': status,
            'running_instances': len(running_processes),
            'processes': running_processes
        }
    
    async def _disk_space_health_check(self, service: Dict[str, Any]) -> Dict[str, Any]:
        """Perform disk space health check."""
        service_name = service.get('name', 'Unknown')
        path = service.get('path', '/')
        threshold_percent = service.get('threshold_percent', 90)
        
        try:
            usage = psutil.disk_usage(path)
            usage_percent = round((usage.used / usage.total) * 100, 2)
            status = 'healthy' if usage_percent < threshold_percent else 'unhealthy'
            
            return {
                'service': service_name,
                'type': 'disk_space',
                'path': path,
                'status': status,
                'usage_percent': usage_percent,
                'threshold_percent': threshold_percent,
                'total_gb': round(usage.total / (1024**3), 2),
                'used_gb': round(usage.used / (1024**3), 2),
                'free_gb': round(usage.free / (1024**3), 2)
            }
            
        except Exception as e:
            return {
                'service': service_name,
                'type': 'disk_space',
                'path': path,
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _monitor_service(self, service_config: Dict[str, Any],
                              duration_seconds: int = 60,
                              interval_seconds: int = 5) -> Dict[str, Any]:
        """Monitor a service over time."""
        service_name = service_config.get('name', 'Unknown')
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        monitoring_results = {
            'service_name': service_name,
            'monitoring_duration': duration_seconds,
            'monitoring_interval': interval_seconds,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'samples': []
        }
        
        while time.time() < end_time:
            # Perform health check
            health_result = await self._perform_health_check([service_config])
            
            sample = {
                'timestamp': datetime.now().isoformat(),
                'health_result': health_result['service_results'][0] if health_result['service_results'] else None
            }
            
            monitoring_results['samples'].append(sample)
            
            # Wait for next interval
            await asyncio.sleep(interval_seconds)
        
        # Calculate statistics
        healthy_samples = sum(1 for s in monitoring_results['samples'] 
                             if s['health_result'] and s['health_result'].get('status') == 'healthy')
        total_samples = len(monitoring_results['samples'])
        
        monitoring_results.update({
            'end_time': datetime.now().isoformat(),
            'total_samples': total_samples,
            'healthy_samples': healthy_samples,
            'unhealthy_samples': total_samples - healthy_samples,
            'availability_percent': round((healthy_samples / total_samples) * 100, 2) if total_samples > 0 else 0
        })
        
        return monitoring_results
    
    async def _analyze_logs(self, log_file_path: str,
                           pattern: Optional[str] = None,
                           level_filter: Optional[str] = None,
                           max_lines: int = 1000) -> Dict[str, Any]:
        """Analyze log files for patterns and statistics."""
        timestamp = datetime.now().isoformat()
        
        try:
            analysis_results = {
                'log_file': log_file_path,
                'analysis_timestamp': timestamp,
                'total_lines': 0,
                'lines_analyzed': 0,
                'log_levels': defaultdict(int),
                'pattern_matches': [],
                'error_patterns': [],
                'statistics': {}
            }
            
            # Common error patterns
            error_patterns = [
                (r'ERROR', 'error'),
                (r'FATAL', 'fatal'),
                (r'CRITICAL', 'critical'),
                (r'exception', 'exception'),
                (r'failed', 'failed'),
                (r'timeout', 'timeout'),
                (r'connection.*refused', 'connection_refused')
            ]
            
            pattern_counts = defaultdict(int)
            
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                for line_num, line in enumerate(file, 1):
                    analysis_results['total_lines'] += 1
                    
                    if line_num > max_lines:
                        break
                    
                    analysis_results['lines_analyzed'] += 1
                    line_lower = line.lower()
                    
                    # Extract log level
                    for level in ['debug', 'info', 'warn', 'warning', 'error', 'fatal', 'critical']:
                        if level in line_lower:
                            analysis_results['log_levels'][level.upper()] += 1
                            break
                    
                    # Check for error patterns
                    for pattern_regex, pattern_name in error_patterns:
                        import re
                        if re.search(pattern_regex, line, re.IGNORECASE):
                            pattern_counts[pattern_name] += 1
                    
                    # Check for custom pattern
                    if pattern:
                        import re
                        if re.search(pattern, line, re.IGNORECASE):
                            analysis_results['pattern_matches'].append({
                                'line_number': line_num,
                                'line_content': line.strip()[:200],  # Limit line length
                                'pattern': pattern
                            })
            
            # Convert defaultdicts to regular dicts
            analysis_results['log_levels'] = dict(analysis_results['log_levels'])
            analysis_results['error_patterns'] = [
                {'pattern': pattern, 'count': count}
                for pattern, count in pattern_counts.items()
            ]
            
            # Calculate statistics
            total_log_entries = sum(analysis_results['log_levels'].values())
            analysis_results['statistics'] = {
                'total_log_entries': total_log_entries,
                'error_rate': round(
                    (analysis_results['log_levels'].get('ERROR', 0) + 
                     analysis_results['log_levels'].get('FATAL', 0) + 
                     analysis_results['log_levels'].get('CRITICAL', 0)) / total_log_entries * 100, 2
                ) if total_log_entries > 0 else 0,
                'warning_rate': round(
                    (analysis_results['log_levels'].get('WARN', 0) + 
                     analysis_results['log_levels'].get('WARNING', 0)) / total_log_entries * 100, 2
                ) if total_log_entries > 0 else 0
            }
            
            return analysis_results
            
        except FileNotFoundError:
            return {
                'log_file': log_file_path,
                'analysis_timestamp': timestamp,
                'error': 'Log file not found'
            }
        except Exception as e:
            return {
                'log_file': log_file_path,
                'analysis_timestamp': timestamp,
                'error': str(e)
            }
    
    async def _generate_performance_report(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report_start = datetime.now()
        
        # Collect metrics multiple times
        samples = []
        for i in range(duration_minutes):
            sample = {
                'timestamp': datetime.now().isoformat(),
                'system': await self._get_system_metrics(),
                'memory': await self._get_memory_metrics(),
                'cpu': await self._get_cpu_metrics(),
                'disk': await self._get_disk_metrics(),
                'network': await self._get_network_metrics()
            }
            samples.append(sample)
            
            if i < duration_minutes - 1:  # Don't sleep after last sample
                await asyncio.sleep(60)  # Wait 1 minute between samples
        
        # Analyze trends
        cpu_usage_values = [sample['system']['result']['cpu']['usage_percent'] for sample in samples]
        memory_usage_values = [sample['system']['result']['memory']['usage_percent'] for sample in samples]
        
        report = {
            'report_period': {
                'start_time': report_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': duration_minutes,
                'samples_collected': len(samples)
            },
            'performance_summary': {
                'cpu': {
                    'average_usage': round(statistics.mean(cpu_usage_values), 2),
                    'peak_usage': max(cpu_usage_values),
                    'min_usage': min(cpu_usage_values),
                    'usage_trend': 'increasing' if cpu_usage_values[-1] > cpu_usage_values[0] else 'decreasing'
                },
                'memory': {
                    'average_usage': round(statistics.mean(memory_usage_values), 2),
                    'peak_usage': max(memory_usage_values),
                    'min_usage': min(memory_usage_values),
                    'usage_trend': 'increasing' if memory_usage_values[-1] > memory_usage_values[0] else 'decreasing'
                }
            },
            'detailed_samples': samples,
            'recommendations': []
        }
        
        # Generate recommendations
        avg_cpu = report['performance_summary']['cpu']['average_usage']
        avg_memory = report['performance_summary']['memory']['average_usage']
        
        if avg_cpu > 80:
            report['recommendations'].append({
                'type': 'cpu',
                'severity': 'high',
                'message': f'High CPU usage detected ({avg_cpu:.1f}%). Consider optimizing processes or adding CPU resources.'
            })
        
        if avg_memory > 90:
            report['recommendations'].append({
                'type': 'memory',
                'severity': 'high',
                'message': f'High memory usage detected ({avg_memory:.1f}%). Consider freeing memory or adding RAM.'
            })
        
        if report['performance_summary']['cpu']['usage_trend'] == 'increasing':
            report['recommendations'].append({
                'type': 'cpu',
                'severity': 'medium',
                'message': 'CPU usage is trending upward. Monitor for potential issues.'
            })
        
        return report
    
    async def _start_continuous_monitoring(self, interval_seconds: int = 60) -> Dict[str, Any]:
        """Start continuous monitoring in background."""
        if self.monitoring_active:
            return {'status': 'already_running'}
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Collect metrics (synchronous version needed for thread)
                    metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'cpu_percent': psutil.cpu_percent(),
                        'memory_percent': psutil.virtual_memory().percent,
                        'disk_percent': psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100
                    }
                    
                    # Store in history
                    self.metric_history['continuous_monitoring'].append(metrics)
                    
                    # Check alert conditions
                    self._check_alert_conditions_sync(metrics)
                    
                except Exception as e:
                    self.logger.error(f"Error in continuous monitoring: {e}")
                
                time.sleep(interval_seconds)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        return {
            'status': 'started',
            'interval_seconds': interval_seconds,
            'start_time': datetime.now().isoformat()
        }
    
    async def _stop_continuous_monitoring(self) -> Dict[str, Any]:
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return {'status': 'not_running'}
        
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        return {
            'status': 'stopped',
            'stop_time': datetime.now().isoformat(),
            'total_samples_collected': len(self.metric_history['continuous_monitoring'])
        }
    
    async def _set_alert_rule(self, alert_id: str,
                             metric: str,
                             operator: str,
                             threshold: float,
                             severity: str = 'medium') -> Dict[str, Any]:
        """Set up alert rule for monitoring."""
        self.alert_rules[alert_id] = {
            'metric': metric,
            'operator': operator,
            'threshold': threshold,
            'severity': severity,
            'created_at': datetime.now().isoformat(),
            'triggered_count': 0,
            'last_triggered': None
        }
        
        return {
            'alert_id': alert_id,
            'status': 'created',
            'rule': self.alert_rules[alert_id]
        }
    
    async def _check_alert_conditions(self) -> Dict[str, Any]:
        """Check current metrics against alert rules."""
        # Get current metrics
        current_metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100
        }
        
        triggered_alerts = self._check_alert_conditions_sync(current_metrics)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'triggered_alerts': triggered_alerts,
            'total_alert_rules': len(self.alert_rules),
            'total_triggered': len(triggered_alerts)
        }
    
    def _check_alert_conditions_sync(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Synchronous version for checking alert conditions."""
        triggered_alerts = []
        
        for alert_id, rule in self.alert_rules.items():
            metric_value = metrics.get(rule['metric'])
            if metric_value is None:
                continue
            
            # Check condition
            triggered = False
            if rule['operator'] == 'greater_than' and metric_value > rule['threshold']:
                triggered = True
            elif rule['operator'] == 'less_than' and metric_value < rule['threshold']:
                triggered = True
            elif rule['operator'] == 'equals' and metric_value == rule['threshold']:
                triggered = True
            
            if triggered:
                rule['triggered_count'] += 1
                rule['last_triggered'] = datetime.now().isoformat()
                
                alert_info = {
                    'alert_id': alert_id,
                    'metric': rule['metric'],
                    'current_value': metric_value,
                    'threshold': rule['threshold'],
                    'operator': rule['operator'],
                    'severity': rule['severity'],
                    'triggered_at': rule['last_triggered']
                }
                
                triggered_alerts.append(alert_info)
                self.alerts_triggered.append(alert_info)
        
        return triggered_alerts
