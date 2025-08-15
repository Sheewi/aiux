#!/usr/bin/env python3
"""
Example usage of all 5 microagents in the Specialization Matrix.

This script demonstrates how to use each microagent individually and 
how to combine them for complex workflows.
"""

import asyncio
import sys
import os
import logging

# Add parent directory to path to import microagents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microagents import (
    WebAutomationAgent,
    DataExtractionAgent, 
    ComputerVisionAgent,
    SystemGovernanceAgent,
    APIOrchestrationAgent
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_web_automation():
    """Demonstrate Web Automation Agent capabilities."""
    print("\n🌐 Web Automation Agent Demo")
    print("=" * 50)
    
    try:
        async with WebAutomationAgent(headless=True) as agent:
            # Navigate to a test page
            result = await agent.navigate("https://httpbin.org/html")
            if result["success"]:
                print(f"✅ Navigation successful: {result['title']}")
                
                # Get accessibility tree
                ax_tree = await agent.get_accessibility_tree()
                if ax_tree["success"]:
                    print(f"✅ Accessibility tree captured")
                    
                # Find accessible elements
                buttons = await agent.find_accessible_elements(role="button")
                links = await agent.find_accessible_elements(role="link")
                print(f"✅ Found {len(buttons)} buttons and {len(links)} links")
                
                # Extract comprehensive page data
                page_data = await agent.extract_page_data()
                if page_data["success"]:
                    print(f"✅ Page analysis complete")
                    print(f"   - Interactive elements: {page_data['accessibility']['interactive_elements']}")
                    
            else:
                print(f"❌ Navigation failed: {result.get('error')}")
                
    except Exception as e:
        print(f"❌ Web automation demo failed: {e}")


async def demo_data_extraction():
    """Demonstrate Data Extraction Agent capabilities."""
    print("\n📊 Data Extraction Agent Demo")
    print("=" * 50)
    
    try:
        agent = DataExtractionAgent()
        
        # Extract data from a test URL
        urls = ["https://httpbin.org/html"]
        results = await agent.extract_batch(urls, method="scrapy")
        
        if results and results[0].success:
            print(f"✅ Data extraction successful")
            print(f"   - URL: {results[0].url}")
            print(f"   - Data keys: {list(results[0].data.keys())}")
            print(f"   - Title: {results[0].data.get('title', 'N/A')}")
            
            # Test structured data extraction
            if 'links' in results[0].data:
                print(f"   - Links found: {len(results[0].data['links'])}")
                
        else:
            print(f"❌ Data extraction failed")
            
    except Exception as e:
        print(f"❌ Data extraction demo failed: {e}")


async def demo_computer_vision():
    """Demonstrate Computer Vision Agent capabilities."""
    print("\n👁️ Computer Vision Agent Demo")
    print("=" * 50)
    
    try:
        agent = ComputerVisionAgent()
        
        # Create a simple test image (100x100 blue square)
        import numpy as np
        test_image = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)  # Blue image
        
        if test_image is not None:
            print("✅ Test image created")
            
            # Analyze the image
            analysis = agent.analyze_image(test_image)
            if analysis.success:
                print(f"✅ Image analysis complete")
                print(f"   - Dimensions: {analysis.data['dimensions']}")
                print(f"   - Blur score: {analysis.data['blur_score']:.2f}")
                print(f"   - Edge density: {analysis.data['edges']['edge_density']:.4f}")
                
            # Extract features
            features = agent.extract_features(test_image, method="orb")
            if features.success:
                print(f"✅ Feature extraction complete")
                print(f"   - Features found: {features.data['num_features']}")
                
            # Enhance image
            enhanced = agent.enhance_image(test_image, operations=["contrast", "sharpen"])
            if enhanced.success:
                print(f"✅ Image enhancement complete")
                print(f"   - Operations applied: {enhanced.data['applied_operations']}")
                
        else:
            print("❌ Could not create test image")
            
    except Exception as e:
        print(f"❌ Computer vision demo failed: {e}")


async def demo_system_governance():
    """Demonstrate System Resource Governance Agent capabilities."""
    print("\n⚙️ System Resource Governance Agent Demo")
    print("=" * 50)
    
    try:
        agent = SystemGovernanceAgent(enable_k8s=False)  # Disable K8s for demo
        
        # Initialize agent
        init_result = await agent.initialize()
        if init_result["success"]:
            print("✅ Agent initialized successfully")
            
            # Get system metrics
            metrics = agent.get_system_metrics()
            if metrics:
                print(f"✅ System metrics collected")
                print(f"   - CPU: {metrics.cpu_percent:.1f}%")
                print(f"   - Memory: {metrics.memory_percent:.1f}%")
                print(f"   - Disk: {metrics.disk_percent:.1f}%")
                print(f"   - Processes: {metrics.process_count}")
                
                # Check for alerts
                alerts = agent.check_alerts(metrics)
                if alerts:
                    print(f"⚠️  Active alerts: {len(alerts)}")
                    for alert in alerts[:3]:  # Show first 3 alerts
                        print(f"   - {alert['type']}: {alert['message']}")
                else:
                    print("✅ No active alerts")
                    
                # Get top processes
                top_procs = agent.get_top_processes(5, sort_by="cpu")
                print(f"✅ Top 5 processes by CPU:")
                for i, proc in enumerate(top_procs[:3], 1):
                    print(f"   {i}. {proc.name} (PID: {proc.pid}) - {proc.cpu_percent:.1f}% CPU")
                    
                # Get optimization recommendations
                optimization = await agent.optimize_resources()
                if optimization["success"]:
                    print(f"✅ Optimization analysis complete")
                    print(f"   - Optimization score: {optimization['optimization_score']}/100")
                    print(f"   - Recommendations: {len(optimization['recommendations'])}")
                    
        else:
            print(f"❌ Agent initialization failed: {init_result}")
            
    except Exception as e:
        print(f"❌ System governance demo failed: {e}")


async def demo_api_orchestration():
    """Demonstrate API Orchestration Agent capabilities."""
    print("\n🔗 API Orchestration Agent Demo")
    print("=" * 50)
    
    try:
        async with APIOrchestrationAgent() as agent:
            print("✅ API orchestration agent started")
            
            # Simple HTTP request
            response = await agent.get("https://httpbin.org/get", 
                                     params={"demo": "microagents"})
            if response.success:
                print(f"✅ HTTP GET request successful")
                print(f"   - Status: {response.status_code}")
                print(f"   - Duration: {response.duration:.3f}s")
                
            # Batch requests
            from microagents.api_orchestration import APIRequest
            
            requests = [
                APIRequest("GET", "https://httpbin.org/get"),
                APIRequest("GET", "https://httpbin.org/headers"),
                APIRequest("GET", "https://httpbin.org/user-agent")
            ]
            
            batch_responses = await agent.batch_requests(requests, max_concurrent=3)
            successful_requests = sum(1 for r in batch_responses if hasattr(r, 'success') and r.success)
            print(f"✅ Batch requests completed: {successful_requests}/{len(requests)} successful")
            
            # Health check
            endpoints = [
                "https://httpbin.org/status/200",
                "https://httpbin.org/status/404",
                "https://httpbin.org/delay/1"
            ]
            
            health = await agent.health_check(endpoints)
            print(f"✅ Health check completed")
            print(f"   - Healthy endpoints: {health['healthy_count']}/{health['total_count']}")
            
            # Demonstrate workflow orchestration
            workflow = [
                {
                    "id": "step1",
                    "type": "http",
                    "request": {
                        "method": "GET",
                        "url": "https://httpbin.org/uuid"
                    }
                },
                {
                    "id": "step2", 
                    "type": "delay",
                    "seconds": 0.5
                },
                {
                    "id": "step3",
                    "type": "http",
                    "request": {
                        "method": "GET",
                        "url": "https://httpbin.org/get",
                        "params": {"from_step1": "workflow_demo"}
                    },
                    "dependencies": ["step1", "step2"]
                }
            ]
            
            workflow_results = await agent.orchestrate_workflow(workflow)
            successful_steps = sum(1 for r in workflow_results if r.get('success'))
            print(f"✅ Workflow orchestration completed: {successful_steps}/{len(workflow)} steps successful")
            
    except Exception as e:
        print(f"❌ API orchestration demo failed: {e}")


async def demo_integrated_workflow():
    """Demonstrate how microagents can work together."""
    print("\n🚀 Integrated Microagents Workflow Demo")
    print("=" * 50)
    
    try:
        # Step 1: Use System Governance Agent to check system resources
        print("Step 1: Checking system resources...")
        governance_agent = SystemGovernanceAgent(enable_k8s=False)
        await governance_agent.initialize()
        
        metrics = governance_agent.get_system_metrics()
        if metrics and metrics.cpu_percent < 80:  # Only proceed if system isn't overloaded
            print(f"✅ System resources OK (CPU: {metrics.cpu_percent:.1f}%)")
            
            # Step 2: Use API Orchestration Agent to fetch a webpage
            print("Step 2: Fetching webpage data...")
            async with APIOrchestrationAgent() as api_agent:
                response = await api_agent.get("https://httpbin.org/html")
                
                if response.success:
                    print("✅ Webpage fetched successfully")
                    
                    # Step 3: Use Data Extraction Agent to parse the content
                    print("Step 3: Extracting structured data...")
                    data_agent = DataExtractionAgent()
                    
                    # Simulate extraction from the fetched content
                    extraction_results = await data_agent.extract_batch(
                        ["https://httpbin.org/html"], 
                        method="scrapy"
                    )
                    
                    if extraction_results and extraction_results[0].success:
                        print("✅ Data extraction completed")
                        
                        # Step 4: Use Web Automation Agent to interact with the page
                        print("Step 4: Automating web interaction...")
                        async with WebAutomationAgent(headless=True) as web_agent:
                            nav_result = await web_agent.navigate("https://httpbin.org/html")
                            
                            if nav_result["success"]:
                                print("✅ Web automation successful")
                                
                                # Step 5: Generate a report using all collected data
                                print("Step 5: Generating integrated report...")
                                
                                report = {
                                    "timestamp": metrics.timestamp.isoformat(),
                                    "system_health": {
                                        "cpu_percent": metrics.cpu_percent,
                                        "memory_percent": metrics.memory_percent,
                                        "status": "healthy" if metrics.cpu_percent < 80 else "warning"
                                    },
                                    "api_performance": {
                                        "response_time": response.duration,
                                        "status_code": response.status_code,
                                        "success": response.success
                                    },
                                    "extracted_data": {
                                        "title": extraction_results[0].data.get("title", "N/A"),
                                        "links_found": len(extraction_results[0].data.get("links", [])),
                                        "extraction_success": extraction_results[0].success
                                    },
                                    "web_automation": {
                                        "page_title": nav_result.get("title", "N/A"),
                                        "navigation_success": nav_result["success"]
                                    }
                                }
                                
                                print("✅ Integrated workflow completed successfully!")
                                print("\n📋 Final Report:")
                                print(f"   - System Status: {report['system_health']['status']}")
                                print(f"   - API Response Time: {report['api_performance']['response_time']:.3f}s")
                                print(f"   - Data Extraction: {report['extracted_data']['extraction_success']}")
                                print(f"   - Web Automation: {report['web_automation']['navigation_success']}")
                                
                                return report
                                
        else:
            print("⚠️  System resources too high, skipping intensive operations")
            
    except Exception as e:
        print(f"❌ Integrated workflow failed: {e}")
        return None


async def main():
    """Main demo function."""
    print("🔥 Microagents Specialization Matrix Demo")
    print("=" * 60)
    print("Demonstrating the 'special forces' squad for each domain:")
    print("• Web Automation: Playwright + AX Tree")
    print("• Data Extraction: Scrapy + Diffbot")  
    print("• Computer Vision: OpenCV + ONNX Runtime")
    print("• System Governance: psutil + K8s API")
    print("• API Orchestration: httpx + GraphQL")
    print("=" * 60)
    
    # Run individual agent demos
    await demo_web_automation()
    await demo_data_extraction()
    await demo_computer_vision()
    await demo_system_governance()
    await demo_api_orchestration()
    
    # Run integrated workflow demo
    await demo_integrated_workflow()
    
    print("\n🎉 All demos completed!")
    print("\nEach microagent is ready for production use with:")
    print("• Comprehensive error handling")
    print("• Async/await support")
    print("• Modular architecture")
    print("• Rich configuration options")
    print("• Performance optimization")


if __name__ == "__main__":
    asyncio.run(main())
