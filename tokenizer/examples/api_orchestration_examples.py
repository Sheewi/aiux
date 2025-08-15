#!/usr/bin/env python3
"""
API Orchestration Agent example using httpx + GraphQL.

This example demonstrates advanced API orchestration capabilities.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microagents.api_orchestration import (
    APIOrchestrationAgent, 
    APIRequest, 
    GraphQLQuery
)


async def http_orchestration_example():
    """Example: Complex HTTP API orchestration."""
    print("🔗 HTTP API Orchestration Example")
    print("-" * 40)
    
    async with APIOrchestrationAgent() as agent:
        # Step 1: Get a UUID
        uuid_response = await agent.get("https://httpbin.org/uuid")
        
        if uuid_response.success:
            uuid_data = uuid_response.data
            print(f"✅ Generated UUID: {uuid_data.get('uuid', 'N/A')[:8]}...")
            
            # Step 2: Use the UUID in a POST request
            post_data = {
                "session_id": uuid_data.get('uuid'),
                "action": "test_workflow",
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            post_response = await agent.post(
                "https://httpbin.org/post",
                json_data=post_data
            )
            
            if post_response.success:
                print("✅ POST request successful")
                response_json = post_response.data.get('json', {})
                print(f"   Session ID: {response_json.get('session_id', 'N/A')[:8]}...")


async def batch_processing_example():
    """Example: Batch API processing with rate limiting."""
    print("\n📊 Batch API Processing Example")
    print("-" * 40)
    
    # Create agent with rate limiting (10 requests per 60 seconds)
    async with APIOrchestrationAgent(rate_limit=10) as agent:
        # Create multiple API requests
        requests = []
        for i in range(5):
            request = APIRequest(
                method="GET",
                url="https://httpbin.org/delay/1",
                params={"request_id": i},
                timeout=5.0
            )
            requests.append(request)
        
        print(f"Processing {len(requests)} requests with rate limiting...")
        
        # Execute batch requests
        responses = await agent.batch_requests(requests, max_concurrent=3)
        
        successful = sum(1 for r in responses if hasattr(r, 'success') and r.success)
        print(f"✅ Batch completed: {successful}/{len(requests)} successful")
        
        # Show timing information
        for i, response in enumerate(responses):
            if hasattr(response, 'duration'):
                print(f"   Request {i+1}: {response.duration:.2f}s")


async def workflow_orchestration_example():
    """Example: Complex multi-step workflow."""
    print("\n🔄 Workflow Orchestration Example")
    print("-" * 40)
    
    async with APIOrchestrationAgent() as agent:
        # Define a complex workflow
        workflow = [
            {
                "id": "get_ip",
                "type": "http",
                "request": {
                    "method": "GET",
                    "url": "https://httpbin.org/ip"
                }
            },
            {
                "id": "wait",
                "type": "delay",
                "seconds": 1
            },
            {
                "id": "get_headers",
                "type": "http",
                "request": {
                    "method": "GET",
                    "url": "https://httpbin.org/headers"
                },
                "dependencies": ["get_ip", "wait"]
            },
            {
                "id": "post_summary",
                "type": "http",
                "request": {
                    "method": "POST",
                    "url": "https://httpbin.org/post",
                    "json_data": {
                        "workflow": "orchestration_demo",
                        "steps_completed": ["get_ip", "wait", "get_headers"],
                        "client_ip": "from_step_1"
                    }
                },
                "dependencies": ["get_headers"]
            }
        ]
        
        print(f"Executing workflow with {len(workflow)} steps...")
        results = await agent.orchestrate_workflow(workflow)
        
        successful_steps = sum(1 for r in results if r.get('success'))
        print(f"✅ Workflow completed: {successful_steps}/{len(workflow)} steps successful")
        
        # Show step results
        for result in results:
            status = "✅" if result.get('success') else "❌"
            duration = result.get('duration', 0)
            print(f"   {status} {result['step_id']}: {duration:.2f}s")


async def graphql_example():
    """Example: GraphQL query execution."""
    print("\n🔍 GraphQL Query Example")
    print("-" * 40)
    
    async with APIOrchestrationAgent() as agent:
        # Add a GraphQL client (using a public GraphQL API)
        graphql_endpoint = "https://countries.trevorblades.com/"
        
        success = await agent.add_graphql_client("countries", graphql_endpoint)
        
        if success:
            print("✅ GraphQL client connected")
            
            # Query for countries
            query = GraphQLQuery(
                query="""
                query GetCountries($filter: CountryFilterInput) {
                    countries(filter: $filter) {
                        code
                        name
                        capital
                        currency
                    }
                }
                """,
                variables={
                    "filter": {
                        "continent": {"eq": "NA"}
                    }
                }
            )
            
            result = await agent.graphql_query("countries", query)
            
            if result.get("success"):
                countries = result["data"].get("countries", [])
                print(f"✅ Found {len(countries)} North American countries")
                
                # Show first few countries
                for country in countries[:3]:
                    print(f"   - {country['name']} ({country['code']})")
            else:
                print(f"❌ GraphQL query failed: {result.get('error')}")
        else:
            print("❌ Failed to connect to GraphQL endpoint")


async def health_monitoring_example():
    """Example: API health monitoring."""
    print("\n🏥 Health Monitoring Example")
    print("-" * 40)
    
    async with APIOrchestrationAgent() as agent:
        # Define endpoints to monitor
        endpoints = [
            "https://httpbin.org/status/200",  # Healthy
            "https://httpbin.org/status/500",  # Error
            "https://httpbin.org/delay/2",     # Slow
            "https://nonexistent-api.example.com"  # Unreachable
        ]
        
        print(f"Monitoring {len(endpoints)} endpoints...")
        
        health_report = await agent.health_check(endpoints)
        
        print(f"Health Report ({health_report['timestamp']}):")
        print(f"✅ Healthy: {health_report['healthy_count']}/{health_report['total_count']}")
        
        for endpoint, status in health_report['endpoints'].items():
            emoji = "✅" if status['status'] == 'healthy' else "❌"
            response_time = status.get('response_time', 0)
            print(f"   {emoji} {endpoint}: {status['status']} ({response_time:.2f}s)")


async def caching_example():
    """Example: Response caching for performance."""
    print("\n💾 Response Caching Example")
    print("-" * 40)
    
    # Create agent with 60-second cache TTL
    async with APIOrchestrationAgent(cache_ttl=60) as agent:
        url = "https://httpbin.org/uuid"
        
        # First request (cache miss)
        print("Making first request (cache miss)...")
        response1 = await agent.get(url)
        
        if response1.success:
            uuid1 = response1.data.get('uuid')
            print(f"✅ First UUID: {uuid1[:8]}... ({response1.duration:.3f}s)")
            
            # Second request (cache hit)
            print("Making second request (cache hit)...")
            response2 = await agent.get(url)
            
            if response2.success:
                uuid2 = response2.data.get('uuid')
                print(f"✅ Second UUID: {uuid2[:8]}... ({response2.duration:.3f}s)")
                
                # Should be the same UUID due to caching
                if uuid1 == uuid2:
                    print("✅ Cache working correctly (same UUID returned)")
                else:
                    print("❌ Cache not working (different UUIDs)")


if __name__ == "__main__":
    async def main():
        print("🔗 API Orchestration Agent Examples")
        print("=" * 50)
        
        try:
            await http_orchestration_example()
            await batch_processing_example()
            await workflow_orchestration_example()
            await graphql_example()
            await health_monitoring_example()
            await caching_example()
            
            print("\n🎉 All API orchestration examples completed!")
            
        except Exception as e:
            print(f"❌ Example failed: {e}")
    
    asyncio.run(main())
