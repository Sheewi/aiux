"""
Hardware Middleware Integration Example
Demonstrates complete hardware orchestration with actionable tokenizers.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any
from pathlib import Path

# Import hardware middleware components
from hardware_middleware import (
    DeviceManager, MessageBus, DeviceDiscoveryService, 
    TelemetryAggregator, CommandValidator,
    CommandValidationRequest, CommandValidationResponse
)

# Import tokenizer components
from tokenizer.action_tokenizer import ActionTokenizer, TokenMode
from tokenizer.microagent_registry import MicroAgentRegistry

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwareOrchestrator:
    """
    Main orchestrator that coordinates hardware middleware and actionable tokenizers.
    This is the central component that the AI system interacts with.
    """
    
    def __init__(self):
        # Initialize core components
        self.device_manager = DeviceManager()
        self.message_bus = MessageBus()
        self.discovery_service = DeviceDiscoveryService()
        self.telemetry_aggregator = TelemetryAggregator()
        self.command_validator = CommandValidator()
        
        # Initialize tokenizer components
        self.microagent_registry = MicroAgentRegistry()
        self.action_tokenizer = ActionTokenizer(self.microagent_registry)
        
        # State tracking
        self.running = False
        self.connected_devices = {}
        self.active_sessions = {}
        
        # Setup event handlers
        self._setup_event_handlers()
        self._setup_approval_system()
    
    def _setup_event_handlers(self):
        """Setup event handlers for device discovery and messaging."""
        
        # Device discovery events
        def on_device_connected(event):
            logger.info(f"Device connected: {event.device_id} ({event.device_type})")
            self._handle_device_connection(event)
        
        def on_device_disconnected(event):
            logger.info(f"Device disconnected: {event.device_id}")
            self._handle_device_disconnection(event)
        
        self.discovery_service.add_event_handler('device_connected', on_device_connected)
        self.discovery_service.add_event_handler('device_disconnected', on_device_disconnected)
        
        # Message bus handlers
        def handle_device_command(message):
            """Handle device command messages."""
            asyncio.create_task(self._process_device_command(message))
        
        def handle_tokenizer_request(message):
            """Handle tokenizer requests."""
            asyncio.create_task(self._process_tokenizer_request(message))
        
        self.message_bus.subscribe('device.command', handle_device_command)
        self.message_bus.subscribe('tokenizer.request', handle_tokenizer_request)
    
    def _setup_approval_system(self):
        """Setup command approval system."""
        
        def approval_callback(request: CommandValidationRequest) -> bool:
            """Handle approval requests for risky commands."""
            logger.warning(f"APPROVAL REQUIRED: {request.command} on {request.device_id}")
            logger.warning(f"Source: {request.source}, Risk assessment needed")
            
            # In a real system, this would involve:
            # - User notification
            # - Security team review
            # - Risk assessment
            # - Manual approval workflow
            
            # For demo purposes, we'll auto-approve certain scenarios
            safe_sources = ['trusted_ai_agent', 'admin_user', 'automated_test']
            if request.source in safe_sources:
                logger.info(f"Auto-approving command from trusted source: {request.source}")
                return True
            
            # Auto-deny from unknown sources
            if request.source == 'unknown':
                logger.warning(f"Auto-denying command from unknown source")
                return False
            
            # For other sources, require manual approval (simulated as denial for demo)
            logger.warning(f"Manual approval required - denying for demo")
            return False
        
        self.command_validator.add_approval_callback(approval_callback)
    
    async def start(self):
        """Start the hardware orchestrator."""
        logger.info("Starting Hardware Orchestrator...")
        
        self.running = True
        
        # Start core services
        await self.message_bus.start()
        await self.discovery_service.start()
        await self.telemetry_aggregator.start()
        
        # Discover and connect to devices
        await self._discover_devices()
        
        logger.info("Hardware Orchestrator started successfully")
    
    async def stop(self):
        """Stop the hardware orchestrator."""
        logger.info("Stopping Hardware Orchestrator...")
        
        self.running = False
        
        # Stop services
        await self.telemetry_aggregator.stop()
        await self.discovery_service.stop()
        await self.message_bus.stop()
        
        # Disconnect devices
        for device_id in list(self.connected_devices.keys()):
            await self.device_manager.disconnect_device(device_id)
        
        logger.info("Hardware Orchestrator stopped")
    
    async def _discover_devices(self):
        """Discover and connect to available devices."""
        logger.info("Discovering available devices...")
        
        devices = self.discovery_service.get_all_devices()
        
        for device_info in devices:
            try:
                # Connect to device through device manager
                device = await self.device_manager.connect_device(
                    device_info['device_id'],
                    device_info['device_type'],
                    device_info.get('connection_params', {})
                )
                
                if device:
                    self.connected_devices[device.device_id] = device
                    
                    # Start telemetry collection for device
                    self.telemetry_aggregator.add_device(device.device_id, device.device_type)
                    
                    logger.info(f"Successfully connected to device: {device.device_id}")
                
            except Exception as e:
                logger.error(f"Failed to connect to device {device_info['device_id']}: {e}")
    
    def _handle_device_connection(self, event):
        """Handle new device connection."""
        # This would trigger automatic device setup, capability detection, etc.
        logger.info(f"Handling connection for device: {event.device_id}")
        
        # Emit notification to AI system
        self.message_bus.publish('system.device_connected', {
            'device_id': event.device_id,
            'device_type': event.device_type,
            'capabilities': event.properties.get('capabilities', []),
            'timestamp': time.time()
        })
    
    def _handle_device_disconnection(self, event):
        """Handle device disconnection."""
        device_id = event.device_id
        
        # Clean up device state
        if device_id in self.connected_devices:
            del self.connected_devices[device_id]
        
        # Remove from telemetry
        self.telemetry_aggregator.remove_device(device_id)
        
        # Emit notification
        self.message_bus.publish('system.device_disconnected', {
            'device_id': device_id,
            'timestamp': time.time()
        })
    
    async def _process_device_command(self, message):
        """Process device command requests."""
        try:
            command_data = message.payload
            device_id = command_data.get('device_id')
            command = command_data.get('command')
            args = command_data.get('args', {})
            source = command_data.get('source', 'unknown')
            
            logger.info(f"Processing command for device {device_id}: {command}")
            
            # Create validation request
            validation_request = CommandValidationRequest(
                command_id=f"cmd_{int(time.time() * 1000)}",
                device_id=device_id,
                command_type='device_action',
                command=command,
                args=args,
                source=source,
                timestamp=time.time()
            )
            
            # Validate command
            validation_result = self.command_validator.validate_command(validation_request)
            
            if validation_result.result.value == 'rejected':
                logger.warning(f"Command rejected: {validation_result.reason}")
                self.message_bus.publish('device.command_result', {
                    'request_id': validation_request.command_id,
                    'success': False,
                    'error': f"Command rejected: {validation_result.reason}",
                    'risk_level': validation_result.risk_level.value
                })
                return
            
            elif validation_result.result.value == 'requires_approval':
                logger.info(f"Command requires approval: {validation_result.approval_message}")
                
                # Request approval
                approved = self.command_validator.request_approval(validation_request)
                
                if not approved:
                    logger.warning(f"Command approval denied")
                    self.message_bus.publish('device.command_result', {
                        'request_id': validation_request.command_id,
                        'success': False,
                        'error': 'Command approval denied',
                        'requires_approval': True
                    })
                    return
                
                logger.info(f"Command approved, proceeding with execution")
            
            # Execute command
            final_command = validation_result.modified_command or command
            final_args = validation_result.modified_args or args
            
            result = await self._execute_device_command(device_id, final_command, final_args)
            
            # Send result
            self.message_bus.publish('device.command_result', {
                'request_id': validation_request.command_id,
                'success': True,
                'result': result,
                'execution_time': time.time()
            })
            
        except Exception as e:
            logger.error(f"Error processing device command: {e}")
            self.message_bus.publish('device.command_result', {
                'request_id': validation_request.command_id if 'validation_request' in locals() else 'unknown',
                'success': False,
                'error': str(e)
            })
    
    async def _execute_device_command(self, device_id: str, command: str, args: Dict[str, Any]) -> Any:
        """Execute a validated command on a device."""
        if device_id not in self.connected_devices:
            raise ValueError(f"Device not connected: {device_id}")
        
        device = self.connected_devices[device_id]
        
        # Execute command based on device type and command
        if command == 'get_status':
            return await device.get_status()
        elif command == 'capture_image' and device.device_type == 'camera':
            return await device.capture_image(**args)
        elif command == 'record_audio' and device.device_type == 'audio':
            return await device.record_audio(**args)
        elif command == 'send_data' and device.device_type in ['usb', 'serial']:
            return await device.send_data(args.get('data', ''))
        else:
            # Generic command execution
            return await device.execute_command(command, **args)
    
    async def _process_tokenizer_request(self, message):
        """Process actionable tokenizer requests."""
        try:
            request_data = message.payload
            mode = TokenMode(request_data.get('mode', 'PRECISE'))
            intent = request_data.get('intent', '')
            context = request_data.get('context', {})
            
            logger.info(f"Processing tokenizer request: {intent} (mode: {mode.value})")
            
            # Generate actionable tokens
            tokens = await self.action_tokenizer.tokenize(intent, context, mode)
            
            # Send result
            self.message_bus.publish('tokenizer.result', {
                'request_id': request_data.get('request_id', 'unknown'),
                'success': True,
                'tokens': [token.to_dict() for token in tokens],
                'execution_time': time.time()
            })
            
        except Exception as e:
            logger.error(f"Error processing tokenizer request: {e}")
            self.message_bus.publish('tokenizer.result', {
                'request_id': request_data.get('request_id', 'unknown'),
                'success': False,
                'error': str(e)
            })
    
    async def execute_ai_command(self, intent: str, context: Dict[str, Any] = None, 
                                source: str = 'ai_system') -> Dict[str, Any]:
        """
        High-level interface for AI systems to execute commands.
        This is the main entry point for AI agents.
        """
        context = context or {}
        request_id = f"ai_req_{int(time.time() * 1000)}"
        
        logger.info(f"AI Command Request [{request_id}]: {intent}")
        
        try:
            # Step 1: Tokenize the intent using actionable tokenizers
            tokens = await self.action_tokenizer.tokenize(
                intent, context, TokenMode.LLM_GUIDED
            )
            
            results = []
            
            # Step 2: Execute each token
            for token in tokens:
                if token.hardware_device:
                    # Hardware command
                    command_result = await self._execute_hardware_token(token, source)
                    results.append({
                        'token_id': token.token_id,
                        'type': 'hardware',
                        'result': command_result
                    })
                
                elif token.microagent_id:
                    # Microagent command
                    agent_result = await self._execute_microagent_token(token, source)
                    results.append({
                        'token_id': token.token_id,
                        'type': 'microagent',
                        'result': agent_result
                    })
                
                else:
                    # Generic action
                    results.append({
                        'token_id': token.token_id,
                        'type': 'action',
                        'result': f"Executed: {token.action}"
                    })
            
            return {
                'request_id': request_id,
                'success': True,
                'intent': intent,
                'tokens_generated': len(tokens),
                'results': results,
                'execution_time': time.time()
            }
        
        except Exception as e:
            logger.error(f"Error executing AI command [{request_id}]: {e}")
            return {
                'request_id': request_id,
                'success': False,
                'error': str(e),
                'intent': intent
            }
    
    async def _execute_hardware_token(self, token, source: str) -> Dict[str, Any]:
        """Execute a hardware-targeted token."""
        # Send command via message bus
        self.message_bus.publish('device.command', {
            'device_id': token.hardware_device,
            'command': token.action,
            'args': token.parameters,
            'source': source,
            'token_id': token.token_id
        })
        
        # Wait for result (simplified - in practice would use proper async handling)
        await asyncio.sleep(0.1)  # Simulate execution time
        
        return {
            'device': token.hardware_device,
            'action': token.action,
            'status': 'executed'
        }
    
    async def _execute_microagent_token(self, token, source: str) -> Dict[str, Any]:
        """Execute a microagent-targeted token."""
        # Load microagent details
        agent_info = self.microagent_registry.get_agent(token.microagent_id)
        
        if not agent_info:
            raise ValueError(f"Unknown microagent: {token.microagent_id}")
        
        # Execute microagent (simplified - would involve actual agent instantiation)
        logger.info(f"Executing microagent {token.microagent_id}: {token.action}")
        
        return {
            'microagent': token.microagent_id,
            'action': token.action,
            'capabilities': agent_info.get('capabilities', []),
            'status': 'executed'
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            'running': self.running,
            'connected_devices': len(self.connected_devices),
            'active_sessions': len(self.active_sessions),
            'telemetry_sources': len(self.telemetry_aggregator.collectors),
            'validation_stats': self.command_validator.get_stats(),
            'timestamp': time.time()
        }

# Example AI integration
async def demo_ai_integration():
    """Demonstrate AI system integration with hardware orchestrator."""
    
    orchestrator = HardwareOrchestrator()
    
    try:
        # Start the orchestrator
        await orchestrator.start()
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Example AI commands
        ai_commands = [
            "Take a photo with the camera",
            "Check status of all USB devices", 
            "Record 5 seconds of audio",
            "Scan for network vulnerabilities",  # This should require approval
            "Generate a security report using data analysis agent",
            "Monitor system performance for anomalies"
        ]
        
        print("\n" + "="*60)
        print("AI HARDWARE ORCHESTRATION DEMO")
        print("="*60)
        
        for i, command in enumerate(ai_commands, 1):
            print(f"\n[AI Command {i}] {command}")
            print("-" * 40)
            
            result = await orchestrator.execute_ai_command(
                command, 
                context={'priority': 'normal', 'timeout': 30},
                source='trusted_ai_agent'
            )
            
            if result['success']:
                print(f"✅ Success - Generated {result['tokens_generated']} tokens")
                for res in result['results']:
                    print(f"   • {res['type']}: {res['result']}")
            else:
                print(f"❌ Failed: {result['error']}")
            
            await asyncio.sleep(1)  # Pause between commands
        
        # Show system status
        print("\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60)
        status = orchestrator.get_system_status()
        print(json.dumps(status, indent=2))
        
        # Show validation statistics
        print("\n" + "="*60)
        print("SECURITY VALIDATION STATS")
        print("="*60)
        validation_stats = orchestrator.command_validator.get_stats()
        print(json.dumps(validation_stats, indent=2))
        
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    print("Starting Hardware Middleware Integration Demo...")
    asyncio.run(demo_ai_integration())
