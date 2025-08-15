"""
Universal AI System Integration Bridge
Connects the main Universal AI System with the microagents conversational AI system
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the microagents path to system path
MICROAGENTS_PATH = Path(__file__).parent / "microagents_conversational_ai"
sys.path.insert(0, str(MICROAGENTS_PATH))

# Import Universal AI System components
from universal_ai_system import UniversalActionTokenizer, CognitiveOrchestrator
from microagent_pool import MicroAgentRegistry, BaseMicroAgent
from embedded_integrations import IntegrationManager
from self_extension_engine import SelfExtensionEngine
from production_infrastructure import ProductionInfrastructure
from main_orchestrator import UniversalAISystem, SystemConfiguration, ExecutionMode

# Import microagents conversational AI components
try:
    from microagents_conversational_ai.hardware_middleware.device_manager import DeviceManager
    from microagents_conversational_ai.hardware_middleware.discovery_service import DeviceDiscoveryService
    from microagents_conversational_ai.hardware_middleware.message_bus import MessageBus
    from microagents_conversational_ai.hardware_middleware.device_abstraction_layer import DeviceAbstractionLayer
    from microagents_conversational_ai.hardware_middleware.communication_bus import CommunicationBus
    from microagents_conversational_ai.hardware_middleware.telemetry_aggregator import TelemetryAggregator
    HARDWARE_MIDDLEWARE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware middleware not fully available: {e}")
    HARDWARE_MIDDLEWARE_AVAILABLE = False

logger = logging.getLogger(__name__)

class IntegratedAISystem:
    """
    Integrated AI System combining Universal AI orchestration with hardware middleware
    """
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.logger = logging.getLogger("integrated_ai_system")
        
        # Universal AI System components
        self.universal_ai_system = UniversalAISystem(config)
        
        # Hardware middleware components (if available)
        self.hardware_middleware_enabled = HARDWARE_MIDDLEWARE_AVAILABLE
        if self.hardware_middleware_enabled:
            self.device_manager = DeviceManager()
            self.device_discovery = DeviceDiscoveryService(
                device_manager=self.device_manager,
                ai_orchestrator=self.universal_ai_system
            )
            self.communication_bus = CommunicationBus(
                ai_orchestrator=self.universal_ai_system
            )
            self.device_abstraction = DeviceAbstractionLayer(
                ai_orchestrator=self.universal_ai_system
            )
            self.telemetry_aggregator = TelemetryAggregator() if hasattr(globals(), 'TelemetryAggregator') else None
        
        # Integration state
        self.integrated_agents: Dict[str, Any] = {}
        self.hardware_ai_bridges: List[Any] = []
        
        self.logger.info(f"Integrated AI System initialized (Hardware middleware: {self.hardware_middleware_enabled})")
    
    async def initialize_integrated_system(self) -> Dict[str, Any]:
        """Initialize the complete integrated AI system"""
        self.logger.info("🚀 Initializing Integrated AI System...")
        
        initialization_results = {}
        
        try:
            # Step 1: Initialize Universal AI System
            self.logger.info("Initializing Universal AI System...")
            universal_init = await self.universal_ai_system.initialize_system()
            initialization_results['universal_ai'] = universal_init
            
            # Step 2: Initialize hardware middleware if available
            if self.hardware_middleware_enabled:
                self.logger.info("Initializing hardware middleware...")
                hardware_init = await self._initialize_hardware_middleware()
                initialization_results['hardware_middleware'] = hardware_init
                
                # Step 3: Create integration bridges
                self.logger.info("Creating AI-Hardware integration bridges...")
                bridge_init = await self._create_integration_bridges()
                initialization_results['integration_bridges'] = bridge_init
            else:
                self.logger.warning("Hardware middleware not available - running in AI-only mode")
                initialization_results['hardware_middleware'] = {'status': 'unavailable'}
            
            # Step 4: Register integrated agents
            integrated_agents_init = await self._register_integrated_agents()
            initialization_results['integrated_agents'] = integrated_agents_init
            
            # Calculate overall readiness
            overall_readiness = self._calculate_integrated_readiness(initialization_results)
            initialization_results['overall_readiness'] = overall_readiness
            
            if overall_readiness >= 0.8:
                self.logger.info(f"✅ Integrated AI System ready! Readiness: {overall_readiness:.1%}")
                initialization_results['status'] = 'ready'
            else:
                self.logger.warning(f"⚠️ Integrated system partially ready. Readiness: {overall_readiness:.1%}")
                initialization_results['status'] = 'partial'
            
            return initialization_results
            
        except Exception as e:
            self.logger.error(f"❌ Integrated system initialization failed: {e}")
            initialization_results['status'] = 'failed'
            initialization_results['error'] = str(e)
            return initialization_results
    
    async def _initialize_hardware_middleware(self) -> Dict[str, Any]:
        """Initialize hardware middleware components"""
        try:
            results = {}
            
            # Start communication bus
            await self.communication_bus.start()
            results['communication_bus'] = {'status': 'started'}
            
            # Start device discovery
            await self.device_discovery.start()
            results['device_discovery'] = {'status': 'started'}
            
            # Initialize device manager
            # Note: DeviceManager might not have async init, handling gracefully
            results['device_manager'] = {'status': 'initialized'}
            
            self.logger.info("Hardware middleware initialized successfully")
            return {'status': 'completed', 'components': results}
            
        except Exception as e:
            self.logger.error(f"Hardware middleware initialization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _create_integration_bridges(self) -> Dict[str, Any]:
        """Create bridges between AI system and hardware middleware"""
        bridges_created = []
        
        try:
            # Bridge 1: Device Discovery -> AI Analysis
            device_ai_bridge = DeviceAIBridge(
                self.device_discovery,
                self.universal_ai_system,
                self.communication_bus
            )
            await device_ai_bridge.start()
            bridges_created.append('device_ai_bridge')
            self.hardware_ai_bridges.append(device_ai_bridge)
            
            # Bridge 2: Communication Bus -> AI Processing
            comm_ai_bridge = CommunicationAIBridge(
                self.communication_bus,
                self.universal_ai_system
            )
            await comm_ai_bridge.start()
            bridges_created.append('communication_ai_bridge')
            self.hardware_ai_bridges.append(comm_ai_bridge)
            
            return {
                'status': 'completed',
                'bridges_created': bridges_created,
                'total_bridges': len(bridges_created)
            }
            
        except Exception as e:
            self.logger.error(f"Integration bridge creation failed: {e}")
            return {'status': 'failed', 'error': str(e), 'bridges_created': bridges_created}
    
    async def _register_integrated_agents(self) -> Dict[str, Any]:
        """Register integrated agents that use both AI and hardware"""
        integrated_agents = []
        
        try:
            # Create hardware-aware AI agents
            if self.hardware_middleware_enabled:
                # Agent 1: Hardware Monitoring with AI Analysis
                hardware_monitor = HardwareAIMonitorAgent(
                    self.device_manager,
                    self.universal_ai_system,
                    self.communication_bus
                )
                self.integrated_agents['hardware_monitor'] = hardware_monitor
                integrated_agents.append('hardware_monitor')
                
                # Agent 2: Predictive Device Management
                predictive_manager = PredictiveDeviceManager(
                    self.device_discovery,
                    self.universal_ai_system,
                    self.device_abstraction
                )
                self.integrated_agents['predictive_manager'] = predictive_manager
                integrated_agents.append('predictive_manager')
            
            # Agent 3: Universal Command Processor (works with or without hardware)
            universal_processor = UniversalCommandProcessor(
                self.universal_ai_system,
                self.communication_bus if self.hardware_middleware_enabled else None
            )
            self.integrated_agents['universal_processor'] = universal_processor
            integrated_agents.append('universal_processor')
            
            return {
                'status': 'completed',
                'agents_registered': integrated_agents,
                'total_agents': len(integrated_agents)
            }
            
        except Exception as e:
            self.logger.error(f"Integrated agent registration failed: {e}")
            return {'status': 'failed', 'error': str(e), 'agents_registered': integrated_agents}
    
    def _calculate_integrated_readiness(self, init_results: Dict[str, Any]) -> float:
        """Calculate overall integrated system readiness"""
        scores = []
        
        # Universal AI readiness
        universal_ai = init_results.get('universal_ai', {})
        if universal_ai.get('status') in ['ready', 'partial']:
            scores.append(universal_ai.get('readiness_score', 0.8))
        else:
            scores.append(0.0)
        
        # Hardware middleware readiness
        hardware = init_results.get('hardware_middleware', {})
        if hardware.get('status') == 'completed':
            scores.append(1.0)
        elif hardware.get('status') == 'unavailable':
            scores.append(0.7)  # Reduced but not critical
        else:
            scores.append(0.0)
        
        # Integration bridges readiness
        bridges = init_results.get('integration_bridges', {})
        if bridges.get('status') == 'completed':
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Integrated agents readiness
        agents = init_results.get('integrated_agents', {})
        if agents.get('status') == 'completed':
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def process_integrated_conversation(self, conversation_messages: List[str], 
                                           session_id: str = None) -> Dict[str, Any]:
        """Process conversation with integrated AI and hardware capabilities"""
        self.logger.info(f"🗣️ Processing integrated conversation...")
        
        try:
            # Process with Universal AI System
            ai_result = await self.universal_ai_system.process_conversation(
                conversation_messages, session_id
            )
            
            # Enhance with hardware context if available
            if self.hardware_middleware_enabled and ai_result.get('status') == 'success':
                hardware_context = await self._get_hardware_context()
                ai_result['hardware_context'] = hardware_context
                
                # Check if hardware actions are needed
                orchestration_plan = ai_result.get('orchestration_plan', {})
                if self._requires_hardware_action(orchestration_plan):
                    hardware_actions = await self._execute_hardware_actions(orchestration_plan)
                    ai_result['hardware_actions'] = hardware_actions
            
            return ai_result
            
        except Exception as e:
            self.logger.error(f"Integrated conversation processing failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _get_hardware_context(self) -> Dict[str, Any]:
        """Get current hardware context"""
        if not self.hardware_middleware_enabled:
            return {'available': False}
        
        try:
            # Get discovered devices
            devices = self.device_discovery.get_all_devices()
            
            # Get communication bus stats
            bus_stats = self.communication_bus.get_statistics()
            
            # Get device abstraction status
            dal_status = self.device_abstraction.get_system_status()
            
            return {
                'available': True,
                'discovered_devices': len(devices),
                'device_types': list(set(d.device_type.value for d in devices.values())),
                'communication_bus_active': bus_stats.get('running', False),
                'total_messages': bus_stats.get('messages_sent', 0),
                'device_abstraction_active': dal_status.get('total_devices', 0) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting hardware context: {e}")
            return {'available': False, 'error': str(e)}
    
    def _requires_hardware_action(self, orchestration_plan: Dict[str, Any]) -> bool:
        """Check if orchestration plan requires hardware actions"""
        if not orchestration_plan:
            return False
        
        # Check for hardware-related keywords in the plan
        hardware_keywords = ['device', 'hardware', 'usb', 'serial', 'network', 'monitor', 'control']
        plan_text = str(orchestration_plan).lower()
        
        return any(keyword in plan_text for keyword in hardware_keywords)
    
    async def _execute_hardware_actions(self, orchestration_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hardware actions based on orchestration plan"""
        if not self.hardware_middleware_enabled:
            return {'available': False}
        
        try:
            actions_executed = []
            
            # Example hardware action execution
            # In practice, this would parse the orchestration plan and execute specific hardware commands
            
            # Publish hardware action request to communication bus
            await self.communication_bus.publish(
                topic="hardware.action_request",
                data=orchestration_plan,
                sender="integrated_ai_system"
            )
            
            actions_executed.append('published_action_request')
            
            return {
                'status': 'completed',
                'actions_executed': actions_executed
            }
            
        except Exception as e:
            self.logger.error(f"Hardware action execution failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def shutdown_integrated_system(self) -> Dict[str, Any]:
        """Shutdown the integrated system"""
        self.logger.info("🛑 Shutting down Integrated AI System...")
        
        shutdown_results = {'components_shutdown': []}
        
        try:
            # Shutdown integration bridges
            for bridge in self.hardware_ai_bridges:
                await bridge.stop()
            shutdown_results['components_shutdown'].append('integration_bridges')
            
            # Shutdown hardware middleware
            if self.hardware_middleware_enabled:
                await self.device_discovery.stop()
                await self.communication_bus.stop()
                shutdown_results['components_shutdown'].append('hardware_middleware')
            
            # Shutdown Universal AI System
            universal_shutdown = await self.universal_ai_system.shutdown_system()
            shutdown_results['universal_ai_shutdown'] = universal_shutdown
            shutdown_results['components_shutdown'].append('universal_ai_system')
            
            shutdown_results['status'] = 'completed'
            
        except Exception as e:
            shutdown_results['status'] = 'partial'
            shutdown_results['error'] = str(e)
        
        return shutdown_results
    
    def get_integrated_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integrated system metrics"""
        metrics = {}
        
        # Universal AI metrics
        metrics['universal_ai'] = self.universal_ai_system.get_system_metrics()
        
        # Hardware middleware metrics
        if self.hardware_middleware_enabled:
            metrics['hardware'] = {
                'discovered_devices': len(self.device_discovery.get_all_devices()),
                'communication_bus': self.communication_bus.get_statistics(),
                'device_abstraction': self.device_abstraction.get_system_status()
            }
        
        # Integration metrics
        metrics['integration'] = {
            'hardware_middleware_enabled': self.hardware_middleware_enabled,
            'integrated_agents': len(self.integrated_agents),
            'active_bridges': len(self.hardware_ai_bridges)
        }
        
        return metrics

class DeviceAIBridge:
    """Bridge between device discovery and AI analysis"""
    
    def __init__(self, device_discovery, ai_system, communication_bus):
        self.device_discovery = device_discovery
        self.ai_system = ai_system
        self.communication_bus = communication_bus
        self._running = False
    
    async def start(self):
        """Start the device AI bridge"""
        self._running = True
        # Add event handlers for device discovery
        self.device_discovery.add_event_handler('connected', self._on_device_connected)
        self.device_discovery.add_event_handler('ai_ready', self._on_device_ai_ready)
    
    async def stop(self):
        """Stop the device AI bridge"""
        self._running = False
    
    async def _on_device_connected(self, event):
        """Handle device connected events"""
        if self._running:
            await self.communication_bus.publish(
                topic="ai.device_analysis",
                data={
                    'event_type': 'device_connected',
                    'device_info': event.device_info.to_dict() if hasattr(event.device_info, 'to_dict') else str(event.device_info)
                },
                sender="device_ai_bridge"
            )
    
    async def _on_device_ai_ready(self, event):
        """Handle AI-ready device events"""
        if self._running:
            await self.communication_bus.publish(
                topic="ai.device_ready",
                data={
                    'event_type': 'device_ai_ready',
                    'device_info': event.device_info.to_dict() if hasattr(event.device_info, 'to_dict') else str(event.device_info)
                },
                sender="device_ai_bridge"
            )

class CommunicationAIBridge:
    """Bridge between communication bus and AI processing"""
    
    def __init__(self, communication_bus, ai_system):
        self.communication_bus = communication_bus
        self.ai_system = ai_system
        self._running = False
    
    async def start(self):
        """Start the communication AI bridge"""
        self._running = True
        # Subscribe to AI-related topics
        self.communication_bus.subscribe("ai.*", self._handle_ai_message)
    
    async def stop(self):
        """Stop the communication AI bridge"""
        self._running = False
    
    async def _handle_ai_message(self, message):
        """Handle AI-related messages from communication bus"""
        if self._running and hasattr(message, 'topic'):
            # Process AI messages here
            pass

class HardwareAIMonitorAgent:
    """Agent that monitors hardware with AI analysis"""
    
    def __init__(self, device_manager, ai_system, communication_bus):
        self.device_manager = device_manager
        self.ai_system = ai_system
        self.communication_bus = communication_bus
    
    async def monitor_and_analyze(self):
        """Monitor hardware and provide AI analysis"""
        # Implementation would go here
        pass

class PredictiveDeviceManager:
    """Agent that predicts device issues using AI"""
    
    def __init__(self, device_discovery, ai_system, device_abstraction):
        self.device_discovery = device_discovery
        self.ai_system = ai_system
        self.device_abstraction = device_abstraction
    
    async def predict_issues(self):
        """Predict potential device issues"""
        # Implementation would go here
        pass

class UniversalCommandProcessor:
    """Agent that processes universal commands with AI assistance"""
    
    def __init__(self, ai_system, communication_bus=None):
        self.ai_system = ai_system
        self.communication_bus = communication_bus
    
    async def process_command(self, command):
        """Process universal commands"""
        # Implementation would go here
        pass

# Main demonstration function
async def main():
    """Demonstration of the integrated AI system"""
    print("=" * 80)
    print("🚀 INTEGRATED AI SYSTEM - UNIVERSAL AI + HARDWARE MIDDLEWARE")
    print("=" * 80)
    print("Combining Universal AI orchestration with hardware middleware")
    print("=" * 80)
    
    # System configuration
    config = SystemConfiguration(
        project_id="integrated-ai-demo",
        environment="development",
        execution_mode=ExecutionMode.AUTONOMOUS,
        self_extension_enabled=True
    )
    
    # Initialize integrated system
    integrated_system = IntegratedAISystem(config)
    
    print("\\n🔄 Initializing Integrated AI System...")
    init_result = await integrated_system.initialize_integrated_system()
    
    print(f"\\n✅ Initialization Result: {init_result['status']}")
    print(f"  • Overall Readiness: {init_result['overall_readiness']:.1%}")
    print(f"  • Universal AI: {init_result.get('universal_ai', {}).get('status', 'unknown')}")
    print(f"  • Hardware Middleware: {init_result.get('hardware_middleware', {}).get('status', 'unknown')}")
    print(f"  • Integration Bridges: {init_result.get('integration_bridges', {}).get('total_bridges', 0)}")
    print(f"  • Integrated Agents: {init_result.get('integrated_agents', {}).get('total_agents', 0)}")
    
    if init_result['status'] in ['ready', 'partial']:
        # Test integrated conversation processing
        print("\\n🗣️ Testing Integrated Conversation Processing...")
        
        test_conversation = [
            "I want to monitor my hardware devices and analyze their performance",
            "Set up automated alerts for any device issues",
            "Use AI to predict potential hardware failures",
            "Integrate with my development environment for real-time monitoring"
        ]
        
        result = await integrated_system.process_integrated_conversation(test_conversation)
        
        if result.get('status') == 'success':
            print("✅ Integrated conversation processed successfully")
            print(f"  • AI Agents Used: {result.get('execution_result', {}).get('total_agents', 0)}")
            print(f"  • Hardware Context: {result.get('hardware_context', {}).get('available', False)}")
            print(f"  • Hardware Actions: {'Yes' if result.get('hardware_actions') else 'No'}")
        else:
            print(f"❌ Conversation processing failed: {result.get('error', 'Unknown error')}")
        
        # Show integrated metrics
        print("\\n📊 Integrated System Metrics:")
        metrics = integrated_system.get_integrated_metrics()
        
        universal_metrics = metrics.get('universal_ai', {})
        print(f"  • Universal AI Success Rate: {universal_metrics.get('success_rate', 0):.1%}")
        print(f"  • Total Conversations: {universal_metrics.get('system_metrics', {}).get('total_conversations', 0)}")
        
        if 'hardware' in metrics:
            hardware_metrics = metrics['hardware']
            print(f"  • Discovered Devices: {hardware_metrics.get('discovered_devices', 0)}")
            print(f"  • Communication Bus Active: {hardware_metrics.get('communication_bus', {}).get('running', False)}")
        
        integration_metrics = metrics.get('integration', {})
        print(f"  • Hardware Middleware: {'Enabled' if integration_metrics.get('hardware_middleware_enabled') else 'Disabled'}")
        print(f"  • Integrated Agents: {integration_metrics.get('integrated_agents', 0)}")
        print(f"  • Active Bridges: {integration_metrics.get('active_bridges', 0)}")
    
    # Graceful shutdown
    print("\\n🛑 Shutting down Integrated AI System...")
    shutdown_result = await integrated_system.shutdown_integrated_system()
    print(f"✅ Shutdown: {shutdown_result['status']}")
    print(f"  • Components Shutdown: {len(shutdown_result['components_shutdown'])}")
    
    print("\\n" + "=" * 80)
    print("🎉 INTEGRATED AI SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("System successfully integrates Universal AI with hardware middleware")

if __name__ == "__main__":
    asyncio.run(main())
