# Comprehensive Tools System

A complete toolkit of specialized tools for the Microagents Conversational AI system, providing capabilities for web search, API integration, file operations, data processing, security, automation, and monitoring.

## 🚀 Features

### Tool Categories

#### 🔍 Search & Web Tools
- **Tavily Search**: AI-powered web search with real-time results
- **Web Scraping**: Intelligent content extraction from web pages
- **Sitemap Extractor**: Extract and analyze website sitemaps

#### 🌐 API Client Tools
- **REST API**: Comprehensive REST API client with authentication
- **GraphQL**: GraphQL query and mutation support
- **Webhooks**: Webhook handling and processing
- **OpenAI API**: Direct integration with OpenAI services

#### 📁 File Operations Tools
- **File Operations**: Complete file system management
- **Configuration**: Multi-format configuration file handling (JSON, YAML, INI, ENV)

#### 📊 Data Processing Tools
- **Data Processing**: Advanced data transformation, analysis, and cleaning
- **Format Conversion**: Convert between CSV, JSON, XML, Excel formats
- **Statistical Analysis**: Descriptive statistics, correlation analysis, outlier detection

#### 🔒 Security Tools
- **Encryption/Decryption**: Multiple encryption methods with key management
- **Password Management**: Password generation and strength analysis
- **Vulnerability Scanning**: Basic security vulnerability detection
- **SSL Validation**: Certificate validation and security checks

#### ⚙️ Automation Tools
- **Task Scheduling**: Flexible task scheduling with multiple triggers
- **Workflow Management**: Multi-step workflow execution
- **Batch Processing**: Parallel and sequential batch operations
- **Retry Mechanisms**: Intelligent retry with exponential backoff

#### 📈 Monitoring Tools
- **System Metrics**: Comprehensive system performance monitoring
- **Health Checks**: Service health monitoring (HTTP, TCP, Process, Disk)
- **Log Analysis**: Automated log file analysis and pattern detection
- **Performance Reports**: Detailed performance analysis and recommendations

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies
```bash
pip install -r tools/requirements.txt
```

### Core Dependencies
```bash
pip install aiohttp aiofiles requests pandas numpy cryptography PyJWT psutil
```

### Optional Dependencies
```bash
pip install beautifulsoup4 lxml PyYAML schedule openai
```

## 📖 Quick Start

### Basic Usage

```python
from tools import global_manager, global_registry

# List available tools
tools = global_registry.list_tools()
print("Available tools:", tools)

# Execute a single tool
result = await global_manager.execute_tool(
    'tavily_search', 
    'search', 
    query='latest AI developments',
    max_results=5
)

# Get tool information
tool_info = global_registry.get_tool_metadata('web_scraping')
print(f"Tool: {tool_info.name} - {tool_info.description}")
```

### Tool Chain Execution

```python
# Execute multiple tools in sequence
chain_config = [
    {
        'tool': 'tavily_search',
        'operation': 'search',
        'parameters': {'query': 'AI news', 'max_results': 3},
        'context_key': 'search_results'
    },
    {
        'tool': 'data_processing',
        'operation': 'analyze',
        'parameters': {'data': '${search_results.result.articles}'},
        'context_key': 'analysis'
    }
]

result = await global_manager.execute_tool_chain(chain_config)
```

### Parallel Tool Execution

```python
# Execute multiple tools in parallel
parallel_config = [
    {
        'tool': 'monitoring',
        'operation': 'system_metrics',
        'parameters': {}
    },
    {
        'tool': 'security',
        'operation': 'scan_vulnerabilities',
        'parameters': {'target_type': 'url', 'target': 'https://example.com'}
    }
]

result = await global_manager.execute_parallel_tools(parallel_config)
```

## 🔧 Tool Configuration

### Search Tools Configuration

```python
# Tavily Search
await global_manager.execute_tool('tavily_search', 'search', 
    query='artificial intelligence',
    max_results=10,
    include_images=True,
    include_answer=True,
    search_depth='advanced'
)

# Web Scraping
await global_manager.execute_tool('web_scraping', 'scrape',
    url='https://example.com',
    extract_links=True,
    extract_images=True,
    respect_robots_txt=True
)
```

### API Client Configuration

```python
# REST API
await global_manager.execute_tool('rest_api', 'request',
    method='GET',
    url='https://api.example.com/data',
    headers={'Authorization': 'Bearer token'},
    timeout=30
)

# OpenAI API
await global_manager.execute_tool('openai_api', 'chat_completion',
    model='gpt-3.5-turbo',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=100
)
```

### File Operations Configuration

```python
# File Operations
await global_manager.execute_tool('file_operations', 'read',
    file_path='/path/to/file.txt',
    encoding='utf-8'
)

# Configuration Management
await global_manager.execute_tool('configuration', 'read',
    file_path='config.json',
    format='json'
)
```

### Data Processing Configuration

```python
# Data Analysis
await global_manager.execute_tool('data_processing', 'analyze',
    data={'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']},
    analysis_type='summary'
)

# Data Transformation
await global_manager.execute_tool('data_processing', 'transform',
    data=dataframe,
    transformations=[
        {'operation': 'normalize', 'params': {'method': 'minmax'}},
        {'operation': 'encode_categorical', 'params': {'method': 'onehot'}}
    ]
)
```

### Security Configuration

```python
# Encryption
await global_manager.execute_tool('security', 'encrypt',
    data='sensitive information',
    method='fernet',
    password='secure_password'
)

# Password Generation
await global_manager.execute_tool('security', 'generate_password',
    length=16,
    include_symbols=True,
    exclude_ambiguous=True
)
```

### Automation Configuration

```python
# Task Scheduling
await global_manager.execute_tool('automation', 'schedule_task',
    task_id='daily_backup',
    task_type='command',
    schedule_type='daily',
    schedule_config={'time': '02:00'},
    task_config={'command': 'backup.sh'}
)

# Workflow Execution
workflow = {
    'id': 'data_pipeline',
    'steps': [
        {'type': 'command', 'config': {'command': 'extract_data.py'}},
        {'type': 'delay', 'config': {'seconds': 5}},
        {'type': 'command', 'config': {'command': 'process_data.py'}}
    ]
}
await global_manager.execute_tool('automation', 'run_workflow',
    workflow_definition=workflow
)
```

### Monitoring Configuration

```python
# System Metrics
await global_manager.execute_tool('monitoring', 'system_metrics')

# Health Checks
services = [
    {'name': 'web_server', 'type': 'http', 'url': 'http://localhost:8080/health'},
    {'name': 'database', 'type': 'tcp', 'host': 'localhost', 'port': 5432},
    {'name': 'nginx', 'type': 'process', 'process_name': 'nginx'}
]
await global_manager.execute_tool('monitoring', 'health_check',
    services=services
)
```

## 🏗️ Architecture

### Tool Structure

```
tools/
├── __init__.py                 # Package initialization and exports
├── requirements.txt           # Dependencies
├── base_tool.py              # Base tool class and interfaces
├── registry.py               # Tool registry and manager
├── tavily_search.py          # Web search tools
├── web_scraping.py           # Web scraping tools
├── api_clients.py            # API integration tools
├── file_operations.py        # File system tools
├── data_processing.py        # Data processing tools
├── security.py               # Security tools
├── automation.py             # Automation tools
└── monitoring.py             # Monitoring tools
```

### Base Tool Interface

All tools inherit from `BaseTool` which provides:
- Standardized execution interface
- Status tracking
- Metadata management
- Error handling
- Rate limiting support

### Registry System

The `ToolsRegistry` provides:
- Tool discovery and registration
- Metadata management
- Configuration validation
- Category organization

### Manager System

The `ToolsManager` provides:
- Tool execution orchestration
- Chain and parallel execution
- Execution history tracking
- Performance monitoring

## 📊 Monitoring and Analytics

### Execution History

```python
# Get execution history
history = global_manager.get_execution_history(limit=10)

# Get tool-specific history
search_history = global_manager.get_execution_history(tool_name='tavily_search')
```

### Performance Statistics

```python
# Get performance stats for all tools
stats = global_manager.get_tool_performance_stats()

# Get registry usage stats
usage_stats = global_registry.get_tool_usage_stats()
```

### Active Tool Monitoring

```python
# Get currently active tools
active = global_manager.get_active_tools()
```

## 🔍 Tool Discovery

### Search Tools

```python
# Search by keyword
search_results = global_registry.search_tools('web')

# List tools by category
web_tools = global_registry.list_tools(category='search_web')

# List tools by capability
api_tools = global_registry.list_tools(capability='api_request')
```

### Tool Information

```python
# Get detailed tool metadata
metadata = global_registry.get_tool_metadata('tavily_search')

# Validate tool configuration
validation = global_registry.validate_tool_config('web_scraping', {
    'url': 'https://example.com',
    'extract_links': True
})
```

## 🚀 Advanced Usage

### Custom Tool Development

```python
from tools.base_tool import BaseTool, ToolStatus, ToolMetadata

class CustomTool(BaseTool):
    async def execute(self, operation: str, **kwargs):
        self.status = ToolStatus.RUNNING
        try:
            # Tool implementation
            result = await self._perform_operation(operation, **kwargs)
            self.status = ToolStatus.COMPLETED
            return result
        except Exception as e:
            self.status = ToolStatus.FAILED
            return {'success': False, 'error': str(e)}
    
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="custom_tool",
            description="Custom tool implementation",
            version="1.0.0",
            capabilities=['custom_operation']
        )

# Register custom tool
global_registry.register_tool_class('custom_tool', CustomTool)
```

### Registry Export/Import

```python
# Export registry for backup or sharing
global_registry.export_registry('tools_registry.json')

# Import external tools
global_registry.import_registry('external_tools.json')
```

## 🔧 Configuration Management

### Environment Variables

Many tools support configuration through environment variables:

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"

# Default configurations
export TOOLS_MAX_TIMEOUT=300
export TOOLS_DEFAULT_RATE_LIMIT=100
```

### Configuration Files

Tools can be configured via JSON/YAML files:

```json
{
  "tools": {
    "tavily_search": {
      "api_key": "your-key",
      "default_max_results": 10
    },
    "web_scraping": {
      "respect_robots_txt": true,
      "default_timeout": 30
    }
  }
}
```

## 🐛 Error Handling

### Tool Execution Errors

```python
try:
    result = await global_manager.execute_tool('invalid_tool', 'operation')
except ValueError as e:
    print(f"Tool not found: {e}")

# Check result for execution errors
if not result['success']:
    print(f"Tool execution failed: {result.get('error')}")
```

### Timeout Handling

```python
# Set custom timeout for tool execution
result = await asyncio.wait_for(
    global_manager.execute_tool('long_running_tool', 'operation'),
    timeout=300  # 5 minutes
)
```

## 📈 Performance Optimization

### Parallel Execution

Use parallel execution for independent operations:

```python
# Instead of sequential execution
results = []
for url in urls:
    result = await global_manager.execute_tool('web_scraping', 'scrape', url=url)
    results.append(result)

# Use parallel execution
parallel_config = [
    {'tool': 'web_scraping', 'operation': 'scrape', 'parameters': {'url': url}}
    for url in urls
]
result = await global_manager.execute_parallel_tools(parallel_config)
```

### Caching and Rate Limiting

Tools implement built-in rate limiting and caching where appropriate:

```python
# Rate limiting is automatically handled
for i in range(100):
    result = await global_manager.execute_tool('api_tool', 'request')
    # Tool will automatically throttle requests
```

## 🔐 Security Considerations

### API Key Management

- Store API keys in environment variables
- Use secure key rotation practices
- Monitor API usage and costs

### Input Validation

- All tools perform input validation
- SQL injection protection in data tools
- Path traversal protection in file tools

### Network Security

- SSL/TLS verification enabled by default
- Certificate validation for HTTPS requests
- Timeout protections against hanging connections

## 🤝 Contributing

### Adding New Tools

1. Create a new tool class inheriting from `BaseTool`
2. Implement the `execute()` method
3. Provide comprehensive metadata
4. Add tests and documentation
5. Register in the registry

### Testing

```bash
# Run tests
pytest tests/

# Run specific tool tests
pytest tests/test_tavily_search.py
```

## 📄 License

This tools system is part of the Microagents Conversational AI project.

## 🆘 Support

For issues, questions, or contributions, please refer to the main project documentation.

---

*Built with ❤️ for the Microagents Conversational AI ecosystem*
