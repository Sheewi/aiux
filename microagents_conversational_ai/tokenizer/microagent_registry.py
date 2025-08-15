"""
MicroAgent Registry - Central registry for all available microagents
Provides capability discovery, metadata management, and agent orchestration.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import importlib.util

logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """Represents a specific capability of a microagent."""
    name: str
    description: str
    input_format: str
    output_format: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AgentMetadata:
    """Metadata for a microagent."""
    agent_id: str
    name: str
    description: str
    version: str
    author: str
    capabilities: List[AgentCapability]
    dependencies: List[str]
    resource_requirements: Dict[str, Any]
    tags: List[str]
    created_at: float
    updated_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['capabilities'] = [cap.to_dict() for cap in self.capabilities]
        return result

class MicroAgentRegistry:
    """
    Central registry for managing and discovering microagents.
    Provides capabilities-based lookup, metadata management, and agent lifecycle.
    """
    
    def __init__(self, registry_file: Optional[str] = None):
        self.registry_file = registry_file or "microagent_registry.json"
        self.agents: Dict[str, AgentMetadata] = {}
        self.capability_index: Dict[str, Set[str]] = {}  # capability -> agent_ids
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> agent_ids
        
        # Load existing registry
        self._load_registry()
        
        # Auto-discover agents if registry is empty
        if not self.agents:
            self._auto_discover_agents()
    
    def _load_registry(self):
        """Load registry from file."""
        try:
            if Path(self.registry_file).exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                for agent_id, agent_data in data.get('agents', {}).items():
                    capabilities = [
                        AgentCapability(**cap) for cap in agent_data.get('capabilities', [])
                    ]
                    agent_data['capabilities'] = capabilities
                    
                    self.agents[agent_id] = AgentMetadata(**agent_data)
                
                self._rebuild_indices()
                logger.info(f"Loaded {len(self.agents)} agents from registry")
        
        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to file."""
        try:
            data = {
                'metadata': {
                    'version': '1.0',
                    'created_at': time.time(),
                    'agent_count': len(self.agents)
                },
                'agents': {
                    agent_id: metadata.to_dict() 
                    for agent_id, metadata in self.agents.items()
                }
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved registry with {len(self.agents)} agents")
        
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _rebuild_indices(self):
        """Rebuild capability and tag indices."""
        self.capability_index.clear()
        self.tag_index.clear()
        
        for agent_id, metadata in self.agents.items():
            # Index capabilities
            for capability in metadata.capabilities:
                if capability.name not in self.capability_index:
                    self.capability_index[capability.name] = set()
                self.capability_index[capability.name].add(agent_id)
            
            # Index tags
            for tag in metadata.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(agent_id)
    
    def _auto_discover_agents(self):
        """Auto-discover agents from the generated_agents directory."""
        logger.info("Auto-discovering microagents...")
        
        agents_dir = Path("generated_agents")
        if not agents_dir.exists():
            logger.warning("generated_agents directory not found")
            return
        
        discovered_count = 0
        
        for py_file in agents_dir.glob("*.py"):
            if py_file.name in ["__init__.py", "base_agent.py"]:
                continue
            
            try:
                agent_info = self._analyze_agent_file(py_file)
                if agent_info:
                    self.register_agent(agent_info['agent_id'], agent_info)
                    discovered_count += 1
            
            except Exception as e:
                logger.debug(f"Failed to analyze {py_file.name}: {e}")
        
        logger.info(f"Auto-discovered {discovered_count} microagents")
        self._save_registry()
    
    def _analyze_agent_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a Python file to extract agent metadata."""
        
        # Read file content for analysis
        content = file_path.read_text()
        
        # Extract agent name from filename
        agent_name = file_path.stem
        
        # Try to determine capabilities from common patterns
        capabilities = self._extract_capabilities_from_code(content, agent_name)
        
        # Determine tags based on agent name and content
        tags = self._extract_tags_from_content(content, agent_name)
        
        # Create agent metadata
        agent_info = {
            'agent_id': agent_name,
            'name': agent_name.replace('_', ' ').title(),
            'description': self._extract_description(content, agent_name),
            'version': '1.0.0',
            'author': 'auto-generated',
            'capabilities': capabilities,
            'dependencies': self._extract_dependencies(content),
            'resource_requirements': {
                'memory_mb': 50,
                'cpu_cores': 1,
                'network_required': False
            },
            'tags': tags,
            'created_at': time.time(),
            'updated_at': time.time()
        }
        
        return agent_info
    
    def _extract_capabilities_from_code(self, content: str, agent_name: str) -> List[Dict[str, Any]]:
        """Extract capabilities from agent code."""
        capabilities = []
        
        # Common capability patterns based on agent names
        capability_patterns = {
            'scraper': {
                'name': 'web_scraping',
                'description': 'Extract data from web pages',
                'input_format': 'url',
                'output_format': 'json'
            },
            'scanner': {
                'name': 'security_scanning',
                'description': 'Scan for security vulnerabilities',
                'input_format': 'target_specification',
                'output_format': 'vulnerability_report'
            },
            'analyzer': {
                'name': 'data_analysis',
                'description': 'Analyze and process data',
                'input_format': 'dataset',
                'output_format': 'analysis_report'
            },
            'generator': {
                'name': 'content_generation',
                'description': 'Generate content or reports',
                'input_format': 'template_and_data',
                'output_format': 'generated_content'
            },
            'monitor': {
                'name': 'monitoring',
                'description': 'Monitor systems or processes',
                'input_format': 'monitoring_config',
                'output_format': 'status_report'
            },
            'detector': {
                'name': 'detection',
                'description': 'Detect patterns or anomalies',
                'input_format': 'data_stream',
                'output_format': 'detection_results'
            },
            'crawler': {
                'name': 'web_crawling',
                'description': 'Crawl and index web content',
                'input_format': 'crawl_configuration',
                'output_format': 'crawled_data'
            },
            'extractor': {
                'name': 'data_extraction',
                'description': 'Extract structured data',
                'input_format': 'source_data',
                'output_format': 'structured_data'
            }
        }
        
        # Check for capability patterns in agent name
        for pattern, capability_template in capability_patterns.items():
            if pattern in agent_name.lower():
                capability = AgentCapability(
                    name=capability_template['name'],
                    description=capability_template['description'],
                    input_format=capability_template['input_format'],
                    output_format=capability_template['output_format'],
                    parameters={}
                )
                capabilities.append(capability.to_dict())
        
        # If no specific pattern found, create a generic capability
        if not capabilities:
            capability = AgentCapability(
                name='general_purpose',
                description=f'General purpose functionality for {agent_name}',
                input_format='generic_input',
                output_format='generic_output',
                parameters={}
            )
            capabilities.append(capability.to_dict())
        
        return capabilities
    
    def _extract_tags_from_content(self, content: str, agent_name: str) -> List[str]:
        """Extract tags from agent content and name."""
        tags = []
        
        # Tags based on common keywords
        tag_keywords = {
            'security': ['security', 'vulnerability', 'penetration', 'exploit', 'attack'],
            'web': ['web', 'http', 'html', 'scraping', 'crawling'],
            'data': ['data', 'analysis', 'processing', 'statistics'],
            'automation': ['automation', 'workflow', 'process', 'task'],
            'network': ['network', 'tcp', 'udp', 'socket', 'connection'],
            'file': ['file', 'directory', 'filesystem', 'storage'],
            'monitoring': ['monitor', 'watch', 'observe', 'track'],
            'reporting': ['report', 'document', 'generate', 'export']
        }
        
        content_lower = content.lower()
        name_lower = agent_name.lower()
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in content_lower or keyword in name_lower for keyword in keywords):
                tags.append(tag)
        
        # Add specific tags based on agent name patterns
        if 'dark_web' in agent_name:
            tags.extend(['dark_web', 'security', 'osint'])
        elif 'compliance' in agent_name:
            tags.extend(['compliance', 'audit', 'regulatory'])
        elif 'bot' in agent_name:
            tags.extend(['automation', 'bot', 'interaction'])
        
        return list(set(tags))  # Remove duplicates
    
    def _extract_description(self, content: str, agent_name: str) -> str:
        """Extract description from agent content."""
        
        # Try to find docstring or comment with description
        lines = content.split('\n')
        
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            if '"""' in line or "'''" in line:
                # Found docstring start
                for j in range(i + 1, min(i + 10, len(lines))):
                    if '"""' in lines[j] or "'''" in lines[j]:
                        # Extract text between docstring markers
                        description_lines = lines[i + 1:j]
                        description = ' '.join(line.strip() for line in description_lines).strip()
                        if description:
                            return description
                        break
        
        # Fallback: generate description from agent name
        return f"Microagent for {agent_name.replace('_', ' ')} functionality"
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from import statements."""
        dependencies = []
        
        # Common import patterns
        import_patterns = [
            'import requests',
            'import selenium',
            'import scrapy',
            'import opencv',
            'import numpy',
            'import pandas',
            'import asyncio',
            'import aiohttp',
            'from playwright'
        ]
        
        content_lower = content.lower()
        
        for pattern in import_patterns:
            if pattern in content_lower:
                # Extract the main module name
                if 'import ' in pattern:
                    module = pattern.split('import ')[-1].split()[0]
                    dependencies.append(module)
                elif 'from ' in pattern:
                    module = pattern.split('from ')[-1].split()[0]
                    dependencies.append(module)
        
        return list(set(dependencies))
    
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """Register a new microagent."""
        
        # Convert capability dicts to AgentCapability objects if needed
        capabilities = []
        for cap in agent_info.get('capabilities', []):
            if isinstance(cap, dict):
                capabilities.append(AgentCapability(**cap))
            else:
                capabilities.append(cap)
        
        agent_info['capabilities'] = capabilities
        
        # Create metadata object
        metadata = AgentMetadata(**agent_info)
        
        # Store in registry
        self.agents[agent_id] = metadata
        
        # Update indices
        self._rebuild_indices()
        
        logger.info(f"Registered agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent metadata by ID."""
        metadata = self.agents.get(agent_id)
        return metadata.to_dict() if metadata else None
    
    def list_agents(self, tags: Optional[List[str]] = None, 
                   capabilities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List agents, optionally filtered by tags or capabilities."""
        
        if not tags and not capabilities:
            return [metadata.to_dict() for metadata in self.agents.values()]
        
        matching_agents = set(self.agents.keys())
        
        # Filter by tags
        if tags:
            tag_matches = set()
            for tag in tags:
                tag_matches.update(self.tag_index.get(tag, set()))
            matching_agents &= tag_matches
        
        # Filter by capabilities
        if capabilities:
            capability_matches = set()
            for capability in capabilities:
                capability_matches.update(self.capability_index.get(capability, set()))
            matching_agents &= capability_matches
        
        return [
            self.agents[agent_id].to_dict() 
            for agent_id in matching_agents
        ]
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents that have a specific capability."""
        return list(self.capability_index.get(capability, set()))
    
    def find_agents_by_tag(self, tag: str) -> List[str]:
        """Find agents that have a specific tag."""
        return list(self.tag_index.get(tag, set()))
    
    def get_capabilities(self) -> List[str]:
        """Get all available capabilities."""
        return list(self.capability_index.keys())
    
    def get_tags(self) -> List[str]:
        """Get all available tags."""
        return list(self.tag_index.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            'total_agents': len(self.agents),
            'total_capabilities': len(self.capability_index),
            'total_tags': len(self.tag_index),
            'agents_by_tag': {
                tag: len(agent_ids) 
                for tag, agent_ids in self.tag_index.items()
            },
            'agents_by_capability': {
                capability: len(agent_ids)
                for capability, agent_ids in self.capability_index.items()
            }
        }
    
    def export_registry(self, filename: str = None):
        """Export registry to a file."""
        if not filename:
            filename = f"microagent_registry_export_{int(time.time())}.json"
        
        data = {
            'export_info': {
                'timestamp': time.time(),
                'total_agents': len(self.agents),
                'version': '1.0'
            },
            'agents': {
                agent_id: metadata.to_dict()
                for agent_id, metadata in self.agents.items()
            },
            'statistics': self.get_stats()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported registry to {filename}")

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create registry
    registry = MicroAgentRegistry()
    
    # Test manual registration
    test_agent = {
        'agent_id': 'test_web_scraper',
        'name': 'Test Web Scraper',
        'description': 'A test web scraping agent',
        'version': '1.0.0',
        'author': 'test',
        'capabilities': [
            {
                'name': 'web_scraping',
                'description': 'Extract data from web pages',
                'input_format': 'url',
                'output_format': 'json',
                'parameters': {'selectors': 'css_selectors'}
            }
        ],
        'dependencies': ['requests', 'beautifulsoup4'],
        'resource_requirements': {
            'memory_mb': 100,
            'cpu_cores': 1,
            'network_required': True
        },
        'tags': ['web', 'scraping', 'data'],
        'created_at': time.time(),
        'updated_at': time.time()
    }
    
    registry.register_agent('test_web_scraper', test_agent)
    
    # Test queries
    print("Registry Demo")
    print("=" * 40)
    
    print(f"Total agents: {len(registry.list_agents())}")
    print(f"Available capabilities: {registry.get_capabilities()}")
    print(f"Available tags: {registry.get_tags()}")
    
    # Test capability search
    web_agents = registry.find_agents_by_capability('web_scraping')
    print(f"Web scraping agents: {web_agents}")
    
    # Test tag search
    security_agents = registry.find_agents_by_tag('security')
    print(f"Security agents: {security_agents}")
    
    # Show statistics
    stats = registry.get_stats()
    print("\nRegistry Statistics:")
    print(json.dumps(stats, indent=2))
