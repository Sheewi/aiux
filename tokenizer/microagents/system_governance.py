"""
System Resource Governance Agent using psutil + Kubernetes API

This microagent specializes in system monitoring, resource management,
and container orchestration governance.
"""

import psutil
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import aiohttp
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: List[float]
    process_count: int


@dataclass
class ProcessInfo:
    """Process information."""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    create_time: datetime
    cmdline: List[str]


@dataclass
class PodMetrics:
    """Kubernetes pod metrics."""
    name: str
    namespace: str
    node: str
    status: str
    cpu_requests: str
    cpu_limits: str
    memory_requests: str
    memory_limits: str
    restart_count: int
    age: timedelta


class KubernetesManager:
    """Manager for Kubernetes API interactions."""
    
    def __init__(self, config_file: str = None, in_cluster: bool = False):
        self.config_file = config_file
        self.in_cluster = in_cluster
        self.v1 = None
        self.apps_v1 = None
        self.metrics_v1 = None
        
    def connect(self) -> bool:
        """Connect to Kubernetes cluster."""
        try:
            if self.in_cluster:
                config.load_incluster_config()
            elif self.config_file:
                config.load_kube_config(config_file=self.config_file)
            else:
                config.load_kube_config()
                
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            
            # Test connection
            self.v1.list_namespace()
            logger.info("Connected to Kubernetes cluster")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes: {e}")
            return False
            
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get basic cluster information."""
        if not self.v1:
            return {"success": False, "error": "Not connected to cluster"}
            
        try:
            # Get nodes
            nodes = self.v1.list_node()
            node_info = []
            
            for node in nodes.items:
                node_info.append({
                    "name": node.metadata.name,
                    "status": "Ready" if any(
                        condition.type == "Ready" and condition.status == "True"
                        for condition in node.status.conditions or []
                    ) else "NotReady",
                    "cpu_capacity": node.status.capacity.get("cpu", "unknown"),
                    "memory_capacity": node.status.capacity.get("memory", "unknown"),
                    "os": node.status.node_info.os_image,
                    "kernel": node.status.node_info.kernel_version,
                    "kubelet": node.status.node_info.kubelet_version
                })
                
            # Get namespaces
            namespaces = self.v1.list_namespace()
            namespace_names = [ns.metadata.name for ns in namespaces.items]
            
            return {
                "success": True,
                "cluster": {
                    "nodes": node_info,
                    "node_count": len(node_info),
                    "namespaces": namespace_names,
                    "namespace_count": len(namespace_names)
                }
            }
            
        except ApiException as e:
            logger.error(f"Kubernetes API error: {e}")
            return {"success": False, "error": str(e)}
            
    def get_pod_metrics(self, namespace: str = None) -> List[PodMetrics]:
        """Get metrics for all pods or pods in a specific namespace."""
        if not self.v1:
            return []
            
        try:
            if namespace:
                pods = self.v1.list_namespaced_pod(namespace)
            else:
                pods = self.v1.list_pod_for_all_namespaces()
                
            pod_metrics = []
            
            for pod in pods.items:
                # Calculate age
                created = pod.metadata.creation_timestamp
                age = datetime.now(created.tzinfo) - created
                
                # Get container resource requests/limits
                cpu_requests = "0"
                cpu_limits = "0" 
                memory_requests = "0"
                memory_limits = "0"
                restart_count = 0
                
                if pod.spec.containers:
                    for container in pod.spec.containers:
                        if container.resources:
                            if container.resources.requests:
                                cpu_requests = container.resources.requests.get("cpu", "0")
                                memory_requests = container.resources.requests.get("memory", "0")
                            if container.resources.limits:
                                cpu_limits = container.resources.limits.get("cpu", "0")
                                memory_limits = container.resources.limits.get("memory", "0")
                                
                if pod.status.container_statuses:
                    restart_count = sum(
                        cs.restart_count for cs in pod.status.container_statuses
                    )
                    
                pod_metrics.append(PodMetrics(
                    name=pod.metadata.name,
                    namespace=pod.metadata.namespace,
                    node=pod.spec.node_name or "unknown",
                    status=pod.status.phase,
                    cpu_requests=cpu_requests,
                    cpu_limits=cpu_limits,
                    memory_requests=memory_requests,
                    memory_limits=memory_limits,
                    restart_count=restart_count,
                    age=age
                ))
                
            return pod_metrics
            
        except ApiException as e:
            logger.error(f"Failed to get pod metrics: {e}")
            return []
            
    def scale_deployment(self, name: str, namespace: str, replicas: int) -> Dict[str, Any]:
        """Scale a deployment to specified number of replicas."""
        if not self.apps_v1:
            return {"success": False, "error": "Not connected to cluster"}
            
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
            
            # Update replica count
            deployment.spec.replicas = replicas
            
            # Apply the update
            self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment
            )
            
            return {
                "success": True,
                "deployment": name,
                "namespace": namespace,
                "new_replicas": replicas
            }
            
        except ApiException as e:
            logger.error(f"Failed to scale deployment: {e}")
            return {"success": False, "error": str(e)}


class SystemGovernanceAgent:
    """
    System resource governance agent with comprehensive monitoring and management.
    
    Features:
    - Real-time system monitoring
    - Process management and analysis
    - Kubernetes cluster governance
    - Resource allocation optimization
    - Alert and threshold management
    - Performance trending
    """
    
    def __init__(self, k8s_config: str = None, enable_k8s: bool = True):
        self.k8s_manager = KubernetesManager(k8s_config) if enable_k8s else None
        self.monitoring_active = False
        self.metric_history = []
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0
        }
        self.monitor_thread = None
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the governance agent."""
        results = {"system": True, "kubernetes": False}
        
        # Test system monitoring
        try:
            psutil.cpu_percent(interval=0.1)
            results["system"] = True
        except Exception as e:
            logger.error(f"System monitoring initialization failed: {e}")
            results["system"] = False
            
        # Initialize Kubernetes if enabled
        if self.k8s_manager:
            results["kubernetes"] = self.k8s_manager.connect()
            
        return {
            "success": any(results.values()),
            "components": results,
            "message": "Governance agent initialized"
        }
        
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024**2)
            network_recv_mb = network.bytes_recv / (1024**2)
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]  # Windows doesn't have load average
                
            # Process count
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk.percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                load_average=load_avg,
                process_count=process_count
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return None
            
    def get_top_processes(self, limit: int = 10, 
                         sort_by: str = "cpu") -> List[ProcessInfo]:
        """Get top processes by CPU or memory usage."""
        try:
            processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 
                                           'memory_percent', 'memory_info', 'create_time', 'cmdline']):
                try:
                    pinfo = proc.info
                    
                    # Convert memory to MB
                    memory_mb = pinfo['memory_info'].rss / (1024**2) if pinfo['memory_info'] else 0
                    
                    # Convert create_time to datetime
                    create_time = datetime.fromtimestamp(pinfo['create_time']) if pinfo['create_time'] else datetime.now()
                    
                    processes.append(ProcessInfo(
                        pid=pinfo['pid'],
                        name=pinfo['name'] or 'unknown',
                        status=pinfo['status'] or 'unknown',
                        cpu_percent=pinfo['cpu_percent'] or 0.0,
                        memory_percent=pinfo['memory_percent'] or 0.0,
                        memory_mb=memory_mb,
                        create_time=create_time,
                        cmdline=pinfo['cmdline'] or []
                    ))
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            # Sort processes
            if sort_by == "cpu":
                processes.sort(key=lambda x: x.cpu_percent, reverse=True)
            elif sort_by == "memory":
                processes.sort(key=lambda x: x.memory_percent, reverse=True)
            else:
                processes.sort(key=lambda x: x.cpu_percent, reverse=True)
                
            return processes[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top processes: {e}")
            return []
            
    def check_alerts(self, metrics: SystemMetrics = None) -> List[Dict[str, Any]]:
        """Check if any metrics exceed alert thresholds."""
        if not metrics:
            metrics = self.get_system_metrics()
            
        if not metrics:
            return []
            
        alerts = []
        
        # Check CPU threshold
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "type": "cpu",
                "level": "warning",
                "message": f"CPU usage at {metrics.cpu_percent:.1f}% (threshold: {self.alert_thresholds['cpu_percent']}%)",
                "value": metrics.cpu_percent,
                "threshold": self.alert_thresholds["cpu_percent"],
                "timestamp": metrics.timestamp
            })
            
        # Check memory threshold
        if metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append({
                "type": "memory",
                "level": "warning",
                "message": f"Memory usage at {metrics.memory_percent:.1f}% (threshold: {self.alert_thresholds['memory_percent']}%)",
                "value": metrics.memory_percent,
                "threshold": self.alert_thresholds["memory_percent"],
                "timestamp": metrics.timestamp
            })
            
        # Check disk threshold
        if metrics.disk_percent > self.alert_thresholds["disk_percent"]:
            alerts.append({
                "type": "disk",
                "level": "critical",
                "message": f"Disk usage at {metrics.disk_percent:.1f}% (threshold: {self.alert_thresholds['disk_percent']}%)",
                "value": metrics.disk_percent,
                "threshold": self.alert_thresholds["disk_percent"],
                "timestamp": metrics.timestamp
            })
            
        return alerts
        
    def set_alert_threshold(self, metric: str, threshold: float) -> bool:
        """Set alert threshold for a metric."""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = threshold
            logger.info(f"Set {metric} threshold to {threshold}%")
            return True
        return False
        
    def start_monitoring(self, interval: int = 30, history_limit: int = 1000):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        def monitor_loop():
            while self.monitoring_active:
                try:
                    metrics = self.get_system_metrics()
                    if metrics:
                        # Add to history
                        self.metric_history.append(asdict(metrics))
                        
                        # Limit history size
                        if len(self.metric_history) > history_limit:
                            self.metric_history = self.metric_history[-history_limit:]
                            
                        # Check for alerts
                        alerts = self.check_alerts(metrics)
                        if alerts:
                            for alert in alerts:
                                logger.warning(f"ALERT: {alert['message']}")
                                
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    
                time.sleep(interval)
                
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started system monitoring with {interval}s interval")
        
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped system monitoring")
        
    def get_metric_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metric history for the specified number of hours."""
        if not self.metric_history:
            return []
            
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_history = []
        for metric in self.metric_history:
            # Convert timestamp string back to datetime if needed
            timestamp = metric["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
            if timestamp >= cutoff_time:
                filtered_history.append(metric)
                
        return filtered_history
        
    async def get_kubernetes_status(self) -> Dict[str, Any]:
        """Get Kubernetes cluster status."""
        if not self.k8s_manager or not self.k8s_manager.v1:
            return {"success": False, "error": "Kubernetes not connected"}
            
        try:
            # Get cluster info
            cluster_info = self.k8s_manager.get_cluster_info()
            
            # Get pod metrics
            pod_metrics = self.k8s_manager.get_pod_metrics()
            
            # Aggregate pod statistics
            pod_stats = {
                "total": len(pod_metrics),
                "running": len([p for p in pod_metrics if p.status == "Running"]),
                "pending": len([p for p in pod_metrics if p.status == "Pending"]),
                "failed": len([p for p in pod_metrics if p.status == "Failed"]),
                "succeeded": len([p for p in pod_metrics if p.status == "Succeeded"]),
                "high_restart_count": len([p for p in pod_metrics if p.restart_count > 5])
            }
            
            return {
                "success": True,
                "cluster": cluster_info.get("cluster", {}),
                "pods": {
                    "statistics": pod_stats,
                    "details": [asdict(pm) for pm in pod_metrics[:20]]  # Limit details
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get Kubernetes status: {e}")
            return {"success": False, "error": str(e)}
            
    async def optimize_resources(self) -> Dict[str, Any]:
        """Provide resource optimization recommendations."""
        try:
            metrics = self.get_system_metrics()
            if not metrics:
                return {"success": False, "error": "Could not get system metrics"}
                
            recommendations = []
            
            # CPU recommendations
            if metrics.cpu_percent > 80:
                recommendations.append({
                    "type": "cpu",
                    "priority": "high",
                    "message": "High CPU usage detected. Consider scaling or optimizing workloads.",
                    "actions": ["Scale horizontally", "Optimize CPU-intensive processes", "Add CPU limits to containers"]
                })
            elif metrics.cpu_percent < 20:
                recommendations.append({
                    "type": "cpu",
                    "priority": "low",
                    "message": "Low CPU usage. Consider downscaling to save resources.",
                    "actions": ["Reduce instance size", "Consolidate workloads", "Lower CPU requests"]
                })
                
            # Memory recommendations
            if metrics.memory_percent > 85:
                recommendations.append({
                    "type": "memory",
                    "priority": "high",
                    "message": "High memory usage detected. Risk of OOM errors.",
                    "actions": ["Add more memory", "Optimize memory usage", "Add memory limits", "Enable swap if appropriate"]
                })
                
            # Disk recommendations
            if metrics.disk_percent > 90:
                recommendations.append({
                    "type": "disk",
                    "priority": "critical",
                    "message": "Disk space critically low. Immediate action required.",
                    "actions": ["Clean up old files", "Add more storage", "Enable log rotation", "Archive old data"]
                })
                
            # Process recommendations
            top_procs = self.get_top_processes(5, "cpu")
            if top_procs and top_procs[0].cpu_percent > 50:
                recommendations.append({
                    "type": "process",
                    "priority": "medium",
                    "message": f"Process '{top_procs[0].name}' using {top_procs[0].cpu_percent:.1f}% CPU",
                    "actions": ["Investigate process behavior", "Consider process optimization", "Check for resource leaks"]
                })
                
            return {
                "success": True,
                "current_metrics": asdict(metrics),
                "recommendations": recommendations,
                "optimization_score": self._calculate_optimization_score(metrics)
            }
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            return {"success": False, "error": str(e)}
            
    def _calculate_optimization_score(self, metrics: SystemMetrics) -> float:
        """Calculate overall system optimization score (0-100)."""
        # Ideal ranges: CPU 20-70%, Memory 30-80%, Disk <80%
        cpu_score = 100 - abs(45 - metrics.cpu_percent) * 2  # Optimal around 45%
        memory_score = 100 - abs(55 - metrics.memory_percent) * 1.5  # Optimal around 55%
        disk_score = max(0, 100 - metrics.disk_percent)  # Lower is better
        
        # Normalize scores
        cpu_score = max(0, min(100, cpu_score))
        memory_score = max(0, min(100, memory_score))
        disk_score = max(0, min(100, disk_score))
        
        # Weighted average
        overall_score = (cpu_score * 0.3 + memory_score * 0.4 + disk_score * 0.3)
        return round(overall_score, 1)


# Convenience functions for quick usage
def quick_system_status() -> Dict[str, Any]:
    """Get quick system status overview."""
    agent = SystemGovernanceAgent(enable_k8s=False)
    metrics = agent.get_system_metrics()
    
    if not metrics:
        return {"success": False, "error": "Could not get metrics"}
        
    return {
        "success": True,
        "cpu_percent": metrics.cpu_percent,
        "memory_percent": metrics.memory_percent,
        "disk_percent": metrics.disk_percent,
        "process_count": metrics.process_count,
        "alerts": agent.check_alerts(metrics)
    }


async def quick_resource_check() -> Dict[str, Any]:
    """Quick resource optimization check."""
    agent = SystemGovernanceAgent(enable_k8s=False)
    return await agent.optimize_resources()


if __name__ == "__main__":
    # Example usage
    async def main():
        agent = SystemGovernanceAgent()
        
        # Initialize
        init_result = await agent.initialize()
        print(f"Initialization: {init_result}")
        
        # Get current metrics
        metrics = agent.get_system_metrics()
        if metrics:
            print(f"CPU: {metrics.cpu_percent:.1f}%")
            print(f"Memory: {metrics.memory_percent:.1f}%")
            print(f"Disk: {metrics.disk_percent:.1f}%")
            
        # Get top processes
        top_procs = agent.get_top_processes(5)
        print(f"Top processes: {len(top_procs)}")
        
        # Check for alerts
        alerts = agent.check_alerts(metrics)
        print(f"Active alerts: {len(alerts)}")
        
        # Get optimization recommendations
        optimization = await agent.optimize_resources()
        print(f"Optimization score: {optimization.get('optimization_score', 'N/A')}")
        
    asyncio.run(main())
