"""
Configuration Management System
Centralized configuration management for the microagent ecosystem.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class AgentConfigManager:
    """
    Centralized configuration management for all agents.
    Supports hierarchical configuration with environment-specific overrides.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / "configs"
        self.configs = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all configuration files from the config directory."""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_configs()
        
        # Load global configuration
        global_config_path = self.config_dir / "global_config.yaml"
        if global_config_path.exists():
            with open(global_config_path, 'r') as f:
                self.configs['global'] = yaml.safe_load(f)
        else:
            self.configs['global'] = self._get_default_global_config()
        
        # Load agent-specific configurations
        agent_configs_dir = self.config_dir / "agents"
        if agent_configs_dir.exists():
            for config_file in agent_configs_dir.glob("*.yaml"):
                agent_name = config_file.stem
                with open(config_file, 'r') as f:
                    self.configs[agent_name] = yaml.safe_load(f)
        
        # Load environment-specific overrides
        env = os.getenv('MICROAGENT_ENV', 'development')
        env_config_path = self.config_dir / f"{env}_config.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
                self._merge_configs(self.configs, env_config)
    
    def _create_default_configs(self):
        """Create default configuration files."""
        # Create global config
        global_config = self._get_default_global_config()
        with open(self.config_dir / "global_config.yaml", 'w') as f:
            yaml.dump(global_config, f, default_flow_style=False)
        
        # Create agent configs directory
        (self.config_dir / "agents").mkdir(exist_ok=True)
    
    def _get_default_global_config(self) -> Dict[str, Any]:
        """Get default global configuration."""
        return {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'handlers': ['console', 'file']
            },
            'execution': {
                'default_timeout': 30,
                'max_retries': 3,
                'retry_backoff_factor': 2,
                'max_backoff_time': 60
            },
            'monitoring': {
                'enable_metrics': True,
                'metrics_export_interval': 60,
                'health_check_interval': 30
            },
            'security': {
                'enable_input_validation': True,
                'enable_output_sanitization': True,
                'max_input_size': 10485760,  # 10MB
                'allowed_file_types': ['.txt', '.json', '.csv', '.xml']
            },
            'performance': {
                'enable_caching': True,
                'cache_ttl': 3600,
                'max_cache_size': 1000,
                'enable_async': True
            }
        }
    
    def _merge_configs(self, base: Dict, override: Dict):
        """Merge configuration dictionaries recursively."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get_config(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific agent or global config.
        
        Args:
            agent_name: Name of the agent (None for global config)
            
        Returns:
            Merged configuration dictionary
        """
        if agent_name is None:
            return self.configs.get('global', {}).copy()
        
        # Start with global config
        config = self.configs.get('global', {}).copy()
        
        # Override with agent-specific config
        agent_config = self.configs.get(agent_name, {})
        self._merge_configs(config, agent_config)
        
        return config
    
    def update_config(self, agent_name: str, updates: Dict[str, Any], persist: bool = True):
        """
        Update configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent
            updates: Configuration updates
            persist: Whether to save changes to disk
        """
        if agent_name not in self.configs:
            self.configs[agent_name] = {}
        
        self._merge_configs(self.configs[agent_name], updates)
        
        if persist:
            self._save_agent_config(agent_name)
    
    def _save_agent_config(self, agent_name: str):
        """Save agent configuration to disk."""
        agent_configs_dir = self.config_dir / "agents"
        agent_configs_dir.mkdir(exist_ok=True)
        
        config_path = agent_configs_dir / f"{agent_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.configs[agent_name], f, default_flow_style=False)
    
    def get_agent_names(self) -> List[str]:
        """Get list of all configured agents."""
        return [name for name in self.configs.keys() if name != 'global']
    
    def validate_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Validate configuration for an agent.
        
        Returns:
            Validation result with status and any issues
        """
        config = self.get_config(agent_name)
        issues = []
        
        # Check required fields
        required_fields = ['execution.default_timeout', 'execution.max_retries']
        for field in required_fields:
            if not self._get_nested_value(config, field):
                issues.append(f"Missing required field: {field}")
        
        # Check value ranges
        timeout = self._get_nested_value(config, 'execution.default_timeout')
        if timeout and (timeout < 1 or timeout > 300):
            issues.append("execution.default_timeout must be between 1 and 300 seconds")
        
        max_retries = self._get_nested_value(config, 'execution.max_retries')
        if max_retries and (max_retries < 0 or max_retries > 10):
            issues.append("execution.max_retries must be between 0 and 10")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'config': config
        }
    
    def _get_nested_value(self, config: Dict, key_path: str) -> Any:
        """Get nested value from config using dot notation."""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def export_config(self, agent_name: Optional[str] = None, format: str = 'yaml') -> str:
        """
        Export configuration in specified format.
        
        Args:
            agent_name: Agent name (None for all configs)
            format: 'yaml' or 'json'
            
        Returns:
            Serialized configuration
        """
        if agent_name:
            config = {agent_name: self.get_config(agent_name)}
        else:
            config = self.configs.copy()
        
        if format.lower() == 'json':
            return json.dumps(config, indent=2)
        else:
            return yaml.dump(config, default_flow_style=False)
    
    def reload_configs(self):
        """Reload all configurations from disk."""
        self.configs.clear()
        self._load_configurations()

# Global configuration manager instance
config_manager = AgentConfigManager()
