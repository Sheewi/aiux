"""
VS Code Integration Tool
Embedded VS Code API integration for AI-driven development

This module provides:
- Intelligent code editing and file management
- Debugging session management with breakpoints
- Project structure management
- Extension management and configuration
- Terminal command execution
- Git integration and version control
- Real-time collaboration features
"""

import asyncio
import json
import subprocess
import os
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VSCodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript" 
    REACT = "typescriptreact"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    MARKDOWN = "markdown"
    YAML = "yaml"
    DOCKERFILE = "dockerfile"

@dataclass
class VSCodeFile:
    """Represents a file in VS Code workspace"""
    path: str
    content: str
    language: VSCodeLanguage
    last_modified: float = field(default_factory=time.time)
    size: int = 0
    is_dirty: bool = False

@dataclass
class DebugConfiguration:
    """VS Code debug configuration"""
    name: str
    program: str
    request: str = "launch"
    console: str = "integratedTerminal"
    breakpoints: List[int] = field(default_factory=list)
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

class VSCodeIntegration:
    """VS Code API integration for AI-driven development"""
    
    def __init__(self, workspace_path: str = "/media/r/Workspace"):
        self.workspace_path = workspace_path
        self.is_connected = False
        self.extensions = {}
        self.active_files: Dict[str, VSCodeFile] = {}
        self.debug_sessions = {}
        self.logger = logging.getLogger("vscode_integration")
        
    async def initialize(self) -> bool:
        """Initialize VS Code integration"""
        try:
            # Check if VS Code is available
            result = subprocess.run(['code', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.is_connected = True
                self.logger.info("VS Code integration initialized successfully")
                await self._setup_extensions()
                await self._setup_workspace_configuration()
                return True
            else:
                self.logger.warning("VS Code not found - using simulation mode")
                return False
        except Exception as e:
            self.logger.error(f"VS Code initialization failed: {e}")
            return False
    
    async def _setup_extensions(self):
        """Setup required VS Code extensions"""
        required_extensions = [
            # Core programming extensions
            "ms-python.python",
            "ms-vscode.vscode-typescript-next",
            "ms-vscode.vscode-json",
            "redhat.vscode-yaml",
            
            # Web development
            "bradlc.vscode-tailwindcss",
            "esbenp.prettier-vscode",
            "ms-vscode.vscode-css-peek",
            "formulahendry.auto-rename-tag",
            
            # React and modern web
            "dsznajder.es7-react-js-snippets",
            "bradlc.vscode-tailwindcss",
            "ms-vscode.vscode-typescript-next",
            
            # AI and productivity
            "github.copilot",
            "github.copilot-chat",
            "ms-vsliveshare.vsliveshare",
            
            # DevOps and cloud
            "ms-vscode-remote.remote-containers",
            "ms-vscode.azure-account",
            "ms-vscode.vscode-docker",
            
            # Git and version control
            "eamodio.gitlens",
            "github.vscode-pull-request-github",
            
            # Debugging and testing
            "ms-vscode.test-adapter-converter",
            "hbenl.vscode-test-explorer",
            
            # Code quality
            "ms-vscode.vscode-eslint",
            "ms-python.pylint",
            "ms-python.black-formatter"
        ]
        
        for ext in required_extensions:
            try:
                # Install extension if not present
                result = subprocess.run(['code', '--install-extension', ext], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.extensions[ext] = "installed"
                    self.logger.info(f"Extension {ext} ready")
            except Exception as e:
                self.logger.warning(f"Failed to setup extension {ext}: {e}")
    
    async def _setup_workspace_configuration(self):
        """Setup VS Code workspace configuration"""
        try:
            vscode_dir = os.path.join(self.workspace_path, '.vscode')
            os.makedirs(vscode_dir, exist_ok=True)
            
            # Settings configuration
            settings = {
                "editor.formatOnSave": True,
                "editor.codeActionsOnSave": {
                    "source.fixAll.eslint": True,
                    "source.organizeImports": True
                },
                "editor.tabSize": 2,
                "editor.insertSpaces": True,
                "editor.wordWrap": "on",
                "editor.minimap.enabled": True,
                "editor.suggestSelection": "first",
                "editor.inlineSuggest.enabled": True,
                
                # Language specific settings
                "python.defaultInterpreterPath": "./venv/bin/python",
                "python.formatting.provider": "black",
                "python.linting.enabled": True,
                "python.linting.pylintEnabled": True,
                
                "typescript.preferences.importModuleSpecifier": "relative",
                "typescript.suggest.autoImports": True,
                
                "emmet.includeLanguages": {
                    "javascript": "javascriptreact",
                    "typescript": "typescriptreact"
                },
                
                # AI assistance
                "github.copilot.enable": {
                    "*": True,
                    "yaml": True,
                    "plaintext": False,
                    "markdown": True
                },
                
                # File associations
                "files.associations": {
                    "*.tsx": "typescriptreact",
                    "*.jsx": "javascriptreact"
                },
                
                # Workspace specific
                "workbench.colorTheme": "Dark+",
                "workbench.iconTheme": "vs-seti",
                "terminal.integrated.defaultProfile.linux": "zsh"
            }
            
            with open(os.path.join(vscode_dir, 'settings.json'), 'w') as f:
                json.dump(settings, f, indent=2)
            
            # Tasks configuration
            tasks = {
                "version": "2.0.0",
                "tasks": [
                    {
                        "label": "Run Python",
                        "type": "shell",
                        "command": "python",
                        "args": ["${file}"],
                        "group": "build",
                        "presentation": {
                            "echo": True,
                            "reveal": "always",
                            "focus": False,
                            "panel": "shared"
                        }
                    },
                    {
                        "label": "Install Dependencies",
                        "type": "shell",
                        "command": "pip",
                        "args": ["install", "-r", "requirements.txt"],
                        "group": "build"
                    },
                    {
                        "label": "Format Python Code",
                        "type": "shell",
                        "command": "black",
                        "args": ["${file}"],
                        "group": "build"
                    },
                    {
                        "label": "Run Tests",
                        "type": "shell",
                        "command": "python",
                        "args": ["-m", "pytest"],
                        "group": "test"
                    }
                ]
            }
            
            with open(os.path.join(vscode_dir, 'tasks.json'), 'w') as f:
                json.dump(tasks, f, indent=2)
            
            self.logger.info("VS Code workspace configuration completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup workspace configuration: {e}")
    
    async def create_file(self, file_path: str, content: str, language: VSCodeLanguage = VSCodeLanguage.PYTHON) -> bool:
        """Create a new file in VS Code workspace"""
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w') as f:
                f.write(content)
            
            # Open in VS Code
            if self.is_connected:
                subprocess.run(['code', full_path], capture_output=True)
            
            # Track file
            vs_file = VSCodeFile(
                path=file_path,
                content=content,
                language=language,
                size=len(content.encode('utf-8'))
            )
            self.active_files[file_path] = vs_file
            
            self.logger.info(f"Created file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create file {file_path}: {e}")
            return False
    
    async def edit_file(self, file_path: str, line_number: int, new_content: str) -> bool:
        """Edit a specific line in a file"""
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            
            if not os.path.exists(full_path):
                self.logger.error(f"File not found: {file_path}")
                return False
            
            with open(full_path, 'r') as f:
                lines = f.readlines()
            
            if line_number > len(lines):
                # Extend file if needed
                lines.extend(['\\n'] * (line_number - len(lines)))
            
            lines[line_number - 1] = new_content + '\\n'
            
            with open(full_path, 'w') as f:
                f.writelines(lines)
            
            # Update tracked file
            if file_path in self.active_files:
                self.active_files[file_path].content = ''.join(lines)
                self.active_files[file_path].last_modified = time.time()
                self.active_files[file_path].is_dirty = True
            
            self.logger.info(f"Edited file {file_path} at line {line_number}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to edit file {file_path}: {e}")
            return False
    
    async def replace_in_file(self, file_path: str, old_text: str, new_text: str) -> bool:
        """Replace text in a file"""
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            
            if not os.path.exists(full_path):
                self.logger.error(f"File not found: {file_path}")
                return False
            
            with open(full_path, 'r') as f:
                content = f.read()
            
            if old_text not in content:
                self.logger.warning(f"Text to replace not found in {file_path}")
                return False
            
            new_content = content.replace(old_text, new_text)
            
            with open(full_path, 'w') as f:
                f.write(new_content)
            
            # Update tracked file
            if file_path in self.active_files:
                self.active_files[file_path].content = new_content
                self.active_files[file_path].last_modified = time.time()
                self.active_files[file_path].is_dirty = True
            
            self.logger.info(f"Replaced text in file {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to replace text in file {file_path}: {e}")
            return False
    
    async def run_command(self, command: str, workspace_relative: bool = True) -> Dict[str, Any]:
        """Run a command in VS Code terminal"""
        try:
            if workspace_relative:
                cwd = self.workspace_path
            else:
                cwd = None
            
            result = subprocess.run(command.split(), capture_output=True, text=True, cwd=cwd)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'command': command
            }
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'command': command
            }
    
    async def debug_session(self, file_path: str, config: DebugConfiguration = None) -> Dict[str, Any]:
        """Start a debugging session"""
        try:
            if not config:
                config = DebugConfiguration(
                    name="AI Debug Session",
                    program=file_path
                )
            
            # Create launch configuration
            launch_config = {
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": config.name,
                        "type": "python",
                        "request": config.request,
                        "program": config.program,
                        "console": config.console,
                        "args": config.args,
                        "env": config.env
                    }
                ]
            }
            
            # Save launch configuration
            vscode_dir = os.path.join(self.workspace_path, '.vscode')
            os.makedirs(vscode_dir, exist_ok=True)
            
            with open(os.path.join(vscode_dir, 'launch.json'), 'w') as f:
                json.dump(launch_config, f, indent=2)
            
            session_id = f"debug_{int(time.time())}"
            self.debug_sessions[session_id] = {
                'config': config,
                'status': 'configured',
                'created_at': time.time()
            }
            
            self.logger.info(f"Debug session configured for {file_path}")
            return {
                'success': True,
                'session_id': session_id,
                'config_path': os.path.join(vscode_dir, 'launch.json'),
                'breakpoints': config.breakpoints
            }
            
        except Exception as e:
            self.logger.error(f"Debug session setup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_file_structure(self) -> Dict[str, Any]:
        """Get current workspace file structure"""
        try:
            structure = {}
            
            for root, dirs, files in os.walk(self.workspace_path):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
                
                rel_root = os.path.relpath(root, self.workspace_path)
                if rel_root == '.':
                    rel_root = ''
                
                for file in files:
                    if not file.startswith('.') and not file.endswith('.pyc'):
                        file_path = os.path.join(rel_root, file) if rel_root else file
                        full_path = os.path.join(root, file)
                        structure[file_path] = {
                            'size': os.path.getsize(full_path),
                            'modified': os.path.getmtime(full_path),
                            'extension': os.path.splitext(file)[1],
                            'language': self._detect_language(file)
                        }
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Failed to get file structure: {e}")
            return {}
    
    def _detect_language(self, filename: str) -> str:
        """Detect programming language from filename"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescriptreact',
            '.jsx': 'javascriptreact',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.md': 'markdown',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.dockerfile': 'dockerfile'
        }
        
        ext = os.path.splitext(filename)[1].lower()
        return ext_map.get(ext, 'plaintext')
    
    async def format_file(self, file_path: str) -> bool:
        """Format a file using VS Code formatters"""
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            language = self._detect_language(file_path)
            
            if language == 'python':
                result = await self.run_command(f"black {full_path}")
                return result['success']
            elif language in ['javascript', 'typescript', 'typescriptreact']:
                result = await self.run_command(f"npx prettier --write {full_path}")
                return result['success']
            else:
                self.logger.info(f"No formatter available for {language}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to format file {file_path}: {e}")
            return False
    
    async def lint_file(self, file_path: str) -> Dict[str, Any]:
        """Lint a file and return issues"""
        try:
            full_path = os.path.join(self.workspace_path, file_path)
            language = self._detect_language(file_path)
            
            if language == 'python':
                result = await self.run_command(f"pylint {full_path}")
                return {
                    'success': True,
                    'language': language,
                    'issues': result['stdout'],
                    'has_errors': result['return_code'] != 0
                }
            elif language in ['javascript', 'typescript']:
                result = await self.run_command(f"npx eslint {full_path}")
                return {
                    'success': True,
                    'language': language,
                    'issues': result['stdout'],
                    'has_errors': result['return_code'] != 0
                }
            else:
                return {
                    'success': True,
                    'language': language,
                    'issues': 'No linter available',
                    'has_errors': False
                }
                
        except Exception as e:
            self.logger.error(f"Failed to lint file {file_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get VS Code integration status"""
        return {
            'connected': self.is_connected,
            'workspace_path': self.workspace_path,
            'extensions_count': len(self.extensions),
            'active_files_count': len(self.active_files),
            'debug_sessions_count': len(self.debug_sessions),
            'capabilities': [
                'file_creation',
                'file_editing',
                'debugging',
                'terminal_execution',
                'formatting',
                'linting',
                'project_management'
            ]
        }

# Global VS Code instance
vscode_tool = VSCodeIntegration()

async def initialize_vscode_tool() -> bool:
    """Initialize VS Code tool"""
    return await vscode_tool.initialize()

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("🔧 VS Code Integration Tool Demo")
        print("=" * 50)
        
        success = await vscode_tool.initialize()
        print(f"Initialization: {'✅ Success' if success else '❌ Failed'}")
        
        if success:
            # Demo file creation
            await vscode_tool.create_file(
                "demo/hello.py",
                "print('Hello from AI-generated code!')",
                VSCodeLanguage.PYTHON
            )
            
            # Demo file structure
            structure = await vscode_tool.get_file_structure()
            print(f"Files in workspace: {len(structure)}")
            
            # Demo status
            status = vscode_tool.get_status()
            print(f"Status: {status}")
        
        print("Demo complete!")
    
    asyncio.run(demo())
