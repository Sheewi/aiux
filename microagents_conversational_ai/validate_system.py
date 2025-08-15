#!/usr/bin/env python3
"""
Quick validation test for the complete microagents system.
Verifies all components are properly integrated and functional.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_hardware_middleware():
    """Test hardware middleware components."""
    print("\n🔧 Testing Hardware Middleware...")
    
    try:
        # Test device manager
        from hardware_middleware import DeviceManager
        device_manager = DeviceManager()
        print("✅ DeviceManager imported successfully")
        
        # Test message bus
        from hardware_middleware import MessageBus
        message_bus = MessageBus()
        message_bus.start()  # These are sync methods, not async
        print("✅ MessageBus started successfully")
        message_bus.stop()
        
        # Test discovery service
        from hardware_middleware import DeviceDiscoveryService
        discovery = DeviceDiscoveryService()
        print("✅ DeviceDiscoveryService imported successfully")
        
        # Test telemetry aggregator
        from hardware_middleware import TelemetryAggregator
        telemetry = TelemetryAggregator()
        print("✅ TelemetryAggregator imported successfully")
        
        # Test command validator
        from hardware_middleware import CommandValidator, CommandValidationRequest
        validator = CommandValidator()
        
        # Test validation
        test_request = CommandValidationRequest(
            command_id="test_001",
            device_id="test_device",
            command_type="test",
            command="echo hello",
            args={},
            source="test_suite",
            timestamp=1234567890.0
        )
        
        result = validator.validate_command(test_request)
        print(f"✅ Command validation working - Result: {result.result.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware middleware test failed: {e}")
        return False

async def test_tokenizer_system():
    """Test tokenizer components."""
    print("\n🔤 Testing Tokenizer System...")
    
    try:
        # Test microagent registry
        from tokenizer.microagent_registry import MicroAgentRegistry
        registry = MicroAgentRegistry()
        
        # Test agent registration
        test_agent_info = {
            'name': 'test_agent',
            'capabilities': ['testing'],
            'description': 'Test agent for validation'
        }
        registry.register_agent('test_agent', test_agent_info)
        
        agents = registry.list_agents()
        print(f"✅ MicroAgentRegistry working - {len(agents)} agents registered")
        
        # Test action tokenizer
        from tokenizer.action_tokenizer import ActionTokenizer, TokenMode
        tokenizer = ActionTokenizer(registry)
        
        # Test tokenization
        tokens = await tokenizer.tokenize(
            "test action", 
            context={'test': True}, 
            mode=TokenMode.PRECISE
        )
        print(f"✅ ActionTokenizer working - Generated {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer system test failed: {e}")
        return False

def test_microagent_ecosystem():
    """Test microagent ecosystem."""
    print("\n🤖 Testing Microagent Ecosystem...")
    
    try:
        # Test base agent
        from generated_agents.base_agent import BaseAgent
        print("✅ BaseAgent imported successfully")
        
        # Test a few sample agents
        test_agents = [
            'web_scraper', 'data_analyzer', 'automation_agent',
            'security_scanner', 'report_generator'
        ]
        
        imported_count = 0
        for agent_name in test_agents:
            try:
                module = __import__(f'generated_agents.{agent_name}', fromlist=[agent_name])
                imported_count += 1
            except ImportError:
                pass  # Agent might not exist, that's ok for this test
        
        print(f"✅ Microagent ecosystem accessible - {imported_count} sample agents found")
        
        # Check generated_agents directory
        agents_dir = Path('generated_agents')
        if agents_dir.exists():
            agent_files = list(agents_dir.glob('*.py'))
            print(f"✅ Found {len(agent_files)} agent files in generated_agents/")
        
        return True
        
    except Exception as e:
        print(f"❌ Microagent ecosystem test failed: {e}")
        return False

async def test_integration():
    """Test full system integration."""
    print("\n🔄 Testing System Integration...")
    
    try:
        # Test orchestrator import
        from hardware_orchestrator_demo import HardwareOrchestrator
        orchestrator = HardwareOrchestrator()
        print("✅ HardwareOrchestrator imported successfully")
        
        # Test system status
        status = orchestrator.get_system_status()
        print(f"✅ System status accessible - Running: {status['running']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_dependencies():
    """Test required dependencies."""
    print("\n📦 Testing Dependencies...")
    
    required_modules = [
        'asyncio', 'json', 'logging', 'time', 'pathlib',
        'typing', 'dataclasses', 'enum', 're', 'hashlib'
    ]
    
    optional_modules = [
        ('zmq', 'ZeroMQ support'),
        ('paho.mqtt', 'MQTT support'), 
        ('psutil', 'System monitoring'),
        ('cv2', 'OpenCV camera support'),
        ('serial', 'Serial device support'),
        ('requests', 'HTTP requests'),
        ('aiohttp', 'Async HTTP')
    ]
    
    # Test required modules
    missing_required = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_required.append(module)
    
    if missing_required:
        print(f"❌ Missing required modules: {missing_required}")
        return False
    else:
        print(f"✅ All required modules available ({len(required_modules)} modules)")
    
    # Test optional modules
    available_optional = []
    for module, description in optional_modules:
        try:
            __import__(module)
            available_optional.append((module, description))
        except ImportError:
            pass
    
    print(f"✅ Optional modules available: {len(available_optional)}/{len(optional_modules)}")
    for module, desc in available_optional:
        print(f"   • {module}: {desc}")
    
    return True

def test_file_structure():
    """Test project file structure."""
    print("\n📁 Testing File Structure...")
    
    required_paths = [
        'ai/',
        'generated_agents/',
        'hardware_middleware/',
        'tokenizer/',
        'hardware_orchestrator_demo.py',
        'demo_conversational_ai.py'
    ]
    
    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    if missing_paths:
        print(f"❌ Missing required paths: {missing_paths}")
        return False
    else:
        print(f"✅ All required paths exist ({len(required_paths)} paths)")
    
    # Check hardware middleware files
    hw_files = [
        'hardware_middleware/__init__.py',
        'hardware_middleware/device_manager.py',
        'hardware_middleware/message_bus.py', 
        'hardware_middleware/discovery_service.py',
        'hardware_middleware/telemetry_aggregator.py',
        'hardware_middleware/command_validator.py'
    ]
    
    hw_missing = []
    for file_path in hw_files:
        if not Path(file_path).exists():
            hw_missing.append(file_path)
    
    if hw_missing:
        print(f"❌ Missing hardware middleware files: {hw_missing}")
        return False
    else:
        print(f"✅ All hardware middleware files present ({len(hw_files)} files)")
    
    return True

async def run_all_tests():
    """Run complete test suite."""
    print("🚀 MICROAGENTS SYSTEM VALIDATION")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Hardware Middleware", test_hardware_middleware),
        ("Tokenizer System", test_tokenizer_system),
        ("Microagent Ecosystem", test_microagent_ecosystem),
        ("System Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to run.")
        print("\nNext steps:")
        print("  1. Run the complete demo: python hardware_orchestrator_demo.py")
        print("  2. Try the AI interface: python demo_conversational_ai.py")
        print("  3. Explore individual components in hardware_middleware/")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    # Add current directory to path for imports
    sys.path.insert(0, str(Path.cwd()))
    
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        sys.exit(1)
