import asyncio
import aiofiles
import os
import shutil
import zipfile
import tarfile
import json
import csv
import yaml
import logging
import mimetypes
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import tempfile
import fnmatch

from .base_tool import BaseTool, ToolStatus, ToolMetadata, ToolType, ToolCapability, create_tool_metadata


class FileOperationsTool(BaseTool):
    """Comprehensive file operations tool for reading, writing, and manipulating files."""
    
    def __init__(self, base_directory: Optional[str] = None, 
                 safe_mode: bool = True,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 config: Dict[str, Any] = None):
        """
        Initialize the file operations tool.
        
        Args:
            base_directory: Base directory for file operations (security boundary)
            safe_mode: Enable safety checks to prevent dangerous operations
            max_file_size: Maximum file size in bytes for read operations
            config: Additional configuration
        """
        # Initialize metadata
        metadata = create_tool_metadata(
            tool_id="file_operations",
            name="File Operations",
            description="Comprehensive file system operations including read, write, delete, copy, move, and directory management",
            tool_type=ToolType.FILE_OPERATIONS,
            version="1.0.0",
            author="System",
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.BATCH_PROCESSING,
                ToolCapability.STATEFUL
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["read", "write", "delete", "copy", "move", "list", "info", "exists"]},
                    "file_path": {"type": "string", "description": "Path to the file or directory"},
                    "content": {"type": "string", "description": "Content to write (for write operations)"},
                    "target_path": {"type": "string", "description": "Target path (for copy/move operations)"},
                    "encoding": {"type": "string", "default": "utf-8", "description": "File encoding"},
                    "create_dirs": {"type": "boolean", "default": False, "description": "Create directories if they don't exist"}
                },
                "required": ["operation", "file_path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "result": {"type": "object"},
                    "error": {"type": "string"}
                }
            },
            timeout=60.0,
            supported_formats=["text", "binary", "json"],
            tags=["file", "io", "filesystem", "storage"]
        )
        
        super().__init__(metadata, config)
        
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self.safe_mode = safe_mode
        self.max_file_size = max_file_size
        self.logger = logging.getLogger(__name__)
        
        # Ensure base directory exists
        self.base_directory.mkdir(parents=True, exist_ok=True)
    
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute file operation.
        
        Args:
            operation: Type of operation (read, write, delete, copy, move, etc.)
            **kwargs: Operation-specific parameters
        """
        self.status = ToolStatus.RUNNING
        
        try:
            operation_map = {
                'read': self._read_file,
                'write': self._write_file,
                'delete': self._delete_file,
                'copy': self._copy_file,
                'move': self._move_file,
                'list': self._list_directory,
                'create_dir': self._create_directory,
                'delete_dir': self._delete_directory,
                'compress': self._compress_files,
                'extract': self._extract_archive,
                'hash': self._calculate_hash,
                'info': self._get_file_info,
                'search': self._search_files,
                'watch': self._watch_directory
            }
            
            if operation not in operation_map:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = await operation_map[operation](**kwargs)
            self.status = ToolStatus.COMPLETED
            return {
                'success': True,
                'operation': operation,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.status = ToolStatus.FAILED
            self.logger.error(f"File operation failed: {e}")
            return {
                'success': False,
                'operation': operation,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and resolve file path within safe boundaries."""
        path = Path(file_path)
        
        if not path.is_absolute():
            path = self.base_directory / path
        
        # Resolve to absolute path and check if within base directory
        resolved_path = path.resolve()
        
        if self.safe_mode:
            try:
                resolved_path.relative_to(self.base_directory.resolve())
            except ValueError:
                raise PermissionError(f"Path {resolved_path} is outside base directory")
        
        return resolved_path
    
    async def _read_file(self, file_path: str, encoding: str = 'utf-8',
                        binary_mode: bool = False) -> Dict[str, Any]:
        """Read file content."""
        path = self._validate_path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.stat().st_size > self.max_file_size:
            raise ValueError(f"File too large: {path.stat().st_size} bytes")
        
        try:
            if binary_mode:
                async with aiofiles.open(path, 'rb') as f:
                    content = await f.read()
                    return {
                        'content': base64.b64encode(content).decode('ascii'),
                        'encoding': 'base64',
                        'size': len(content),
                        'path': str(path)
                    }
            else:
                async with aiofiles.open(path, 'r', encoding=encoding) as f:
                    content = await f.read()
                    return {
                        'content': content,
                        'encoding': encoding,
                        'size': len(content),
                        'path': str(path)
                    }
        except UnicodeDecodeError:
            # Fallback to binary mode if text decoding fails
            return await self._read_file(file_path, binary_mode=True)
    
    async def _write_file(self, file_path: str, content: str,
                         encoding: str = 'utf-8', append: bool = False,
                         create_dirs: bool = True) -> Dict[str, Any]:
        """Write content to file."""
        path = self._validate_path(file_path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if append else 'w'
        
        async with aiofiles.open(path, mode, encoding=encoding) as f:
            await f.write(content)
        
        return {
            'path': str(path),
            'size': path.stat().st_size,
            'mode': 'append' if append else 'write'
        }
    
    async def _delete_file(self, file_path: str, force: bool = False) -> Dict[str, Any]:
        """Delete a file."""
        path = self._validate_path(file_path)
        
        if not path.exists():
            if not force:
                raise FileNotFoundError(f"File not found: {path}")
            return {'path': str(path), 'status': 'already_deleted'}
        
        if path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {path}")
        
        path.unlink()
        return {'path': str(path), 'status': 'deleted'}
    
    async def _copy_file(self, source: str, destination: str,
                        overwrite: bool = False) -> Dict[str, Any]:
        """Copy a file."""
        src_path = self._validate_path(source)
        dst_path = self._validate_path(destination)
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {dst_path}")
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        
        return {
            'source': str(src_path),
            'destination': str(dst_path),
            'size': dst_path.stat().st_size
        }
    
    async def _move_file(self, source: str, destination: str,
                        overwrite: bool = False) -> Dict[str, Any]:
        """Move/rename a file."""
        src_path = self._validate_path(source)
        dst_path = self._validate_path(destination)
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {dst_path}")
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        
        return {
            'source': str(src_path),
            'destination': str(dst_path)
        }
    
    async def _list_directory(self, directory: str = ".", 
                             pattern: Optional[str] = None,
                             recursive: bool = False,
                             include_hidden: bool = False) -> Dict[str, Any]:
        """List directory contents."""
        dir_path = self._validate_path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        files = []
        directories = []
        
        if recursive:
            pattern_glob = pattern or "*"
            for item in dir_path.rglob(pattern_glob):
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                item_info = {
                    'name': item.name,
                    'path': str(item.relative_to(dir_path)),
                    'size': item.stat().st_size if item.is_file() else None,
                    'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    'type': 'file' if item.is_file() else 'directory'
                }
                
                if item.is_file():
                    files.append(item_info)
                else:
                    directories.append(item_info)
        else:
            for item in dir_path.iterdir():
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                if pattern and not fnmatch.fnmatch(item.name, pattern):
                    continue
                
                item_info = {
                    'name': item.name,
                    'path': str(item.relative_to(dir_path)),
                    'size': item.stat().st_size if item.is_file() else None,
                    'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    'type': 'file' if item.is_file() else 'directory'
                }
                
                if item.is_file():
                    files.append(item_info)
                else:
                    directories.append(item_info)
        
        return {
            'directory': str(dir_path),
            'files': files,
            'directories': directories,
            'total_files': len(files),
            'total_directories': len(directories)
        }
    
    async def _create_directory(self, directory: str, 
                               parents: bool = True) -> Dict[str, Any]:
        """Create a directory."""
        dir_path = self._validate_path(directory)
        
        if dir_path.exists():
            if dir_path.is_dir():
                return {'path': str(dir_path), 'status': 'already_exists'}
            else:
                raise FileExistsError(f"File exists with same name: {dir_path}")
        
        dir_path.mkdir(parents=parents, exist_ok=True)
        
        return {
            'path': str(dir_path),
            'status': 'created'
        }
    
    async def _delete_directory(self, directory: str, 
                               force: bool = False) -> Dict[str, Any]:
        """Delete a directory."""
        dir_path = self._validate_path(directory)
        
        if not dir_path.exists():
            if not force:
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            return {'path': str(dir_path), 'status': 'already_deleted'}
        
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        if any(dir_path.iterdir()) and not force:
            raise OSError(f"Directory not empty: {dir_path}")
        
        shutil.rmtree(dir_path)
        
        return {
            'path': str(dir_path),
            'status': 'deleted'
        }
    
    async def _compress_files(self, files: List[str], archive_path: str,
                             format: str = 'zip') -> Dict[str, Any]:
        """Compress files into an archive."""
        archive_path = self._validate_path(archive_path)
        
        if format not in ['zip', 'tar', 'tar.gz', 'tar.bz2']:
            raise ValueError(f"Unsupported archive format: {format}")
        
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        compressed_files = []
        total_size = 0
        
        if format == 'zip':
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in files:
                    file_path = self._validate_path(file_path)
                    if file_path.exists():
                        zf.write(file_path, file_path.name)
                        compressed_files.append(str(file_path))
                        total_size += file_path.stat().st_size
        else:
            mode_map = {
                'tar': 'w',
                'tar.gz': 'w:gz',
                'tar.bz2': 'w:bz2'
            }
            
            with tarfile.open(archive_path, mode_map[format]) as tf:
                for file_path in files:
                    file_path = self._validate_path(file_path)
                    if file_path.exists():
                        tf.add(file_path, file_path.name)
                        compressed_files.append(str(file_path))
                        total_size += file_path.stat().st_size
        
        return {
            'archive_path': str(archive_path),
            'format': format,
            'files_compressed': compressed_files,
            'total_files': len(compressed_files),
            'original_size': total_size,
            'compressed_size': archive_path.stat().st_size,
            'compression_ratio': round(archive_path.stat().st_size / total_size, 2) if total_size > 0 else 0
        }
    
    async def _extract_archive(self, archive_path: str, 
                              destination: str = None) -> Dict[str, Any]:
        """Extract files from an archive."""
        archive_path = self._validate_path(archive_path)
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        if destination is None:
            destination = archive_path.parent / archive_path.stem
        else:
            destination = self._validate_path(destination)
        
        destination.mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        
        # Detect archive format
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(destination)
                extracted_files = zf.namelist()
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(destination)
                extracted_files = tf.getnames()
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
        
        return {
            'archive_path': str(archive_path),
            'destination': str(destination),
            'extracted_files': extracted_files,
            'total_extracted': len(extracted_files)
        }
    
    async def _calculate_hash(self, file_path: str, 
                             algorithm: str = 'sha256') -> Dict[str, Any]:
        """Calculate file hash."""
        path = self._validate_path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_obj = hashlib.new(algorithm)
        
        async with aiofiles.open(path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_obj.update(chunk)
        
        return {
            'file_path': str(path),
            'algorithm': algorithm,
            'hash': hash_obj.hexdigest(),
            'file_size': path.stat().st_size
        }
    
    async def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information."""
        path = self._validate_path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        stat = path.stat()
        
        info = {
            'path': str(path),
            'name': path.name,
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'is_file': path.is_file(),
            'is_directory': path.is_dir(),
            'is_symlink': path.is_symlink(),
            'permissions': oct(stat.st_mode)[-3:],
            'owner_uid': stat.st_uid,
            'group_gid': stat.st_gid
        }
        
        if path.is_file():
            mime_type, encoding = mimetypes.guess_type(str(path))
            info.update({
                'mime_type': mime_type,
                'encoding': encoding,
                'extension': path.suffix
            })
        
        return info
    
    async def _search_files(self, directory: str = ".", 
                           pattern: str = "*",
                           content_pattern: Optional[str] = None,
                           recursive: bool = True) -> Dict[str, Any]:
        """Search for files by name and optionally content."""
        dir_path = self._validate_path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        matches = []
        
        search_pattern = f"**/{pattern}" if recursive else pattern
        
        for file_path in dir_path.glob(search_pattern):
            if file_path.is_file():
                match_info = {
                    'path': str(file_path.relative_to(dir_path)),
                    'absolute_path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
                # Search content if pattern provided
                if content_pattern and file_path.stat().st_size < self.max_file_size:
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                            if content_pattern.lower() in content.lower():
                                match_info['content_match'] = True
                                # Find line numbers with matches
                                lines = content.split('\n')
                                match_lines = [
                                    i + 1 for i, line in enumerate(lines)
                                    if content_pattern.lower() in line.lower()
                                ]
                                match_info['match_lines'] = match_lines[:10]  # Limit to 10 lines
                            else:
                                continue  # Skip if content doesn't match
                    except (UnicodeDecodeError, PermissionError):
                        # Skip files that can't be read as text
                        if content_pattern:
                            continue
                
                matches.append(match_info)
        
        return {
            'directory': str(dir_path),
            'pattern': pattern,
            'content_pattern': content_pattern,
            'matches': matches,
            'total_matches': len(matches)
        }
    
    async def _watch_directory(self, directory: str = ".", 
                              duration: int = 30) -> Dict[str, Any]:
        """Watch directory for changes (simplified implementation)."""
        dir_path = self._validate_path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        # Take initial snapshot
        initial_files = {}
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                initial_files[str(file_path)] = file_path.stat().st_mtime
        
        # Wait for specified duration
        await asyncio.sleep(duration)
        
        # Take final snapshot and compare
        final_files = {}
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                final_files[str(file_path)] = file_path.stat().st_mtime
        
        # Detect changes
        added = set(final_files.keys()) - set(initial_files.keys())
        removed = set(initial_files.keys()) - set(final_files.keys())
        modified = {
            path for path in initial_files.keys() & final_files.keys()
            if initial_files[path] != final_files[path]
        }
        
        return {
            'directory': str(dir_path),
            'duration': duration,
            'changes': {
                'added': list(added),
                'removed': list(removed),
                'modified': list(modified)
            },
            'total_changes': len(added) + len(removed) + len(modified)
        }


class ConfigurationTool(BaseTool):
    """Tool for reading and writing configuration files in various formats."""
    
    def __init__(self, base_directory: Optional[str] = None, config: Dict[str, Any] = None):
        # Initialize metadata
        metadata = create_tool_metadata(
            tool_id="configuration",
            name="Configuration Manager",
            description="Tool for reading, writing, and managing configuration files in various formats (JSON, YAML, TOML, INI)",
            tool_type=ToolType.FILE_OPERATIONS,
            version="1.0.0",
            author="System",
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.CACHEABLE
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["read", "write", "merge", "validate"]},
                    "config_path": {"type": "string", "description": "Path to configuration file"},
                    "format": {"type": "string", "enum": ["json", "yaml", "toml", "ini"], "description": "Configuration format"},
                    "data": {"type": "object", "description": "Configuration data to write"}
                },
                "required": ["operation", "config_path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "result": {"type": "object"},
                    "error": {"type": "string"}
                }
            },
            timeout=30.0,
            supported_formats=["json", "yaml", "toml", "ini"],
            tags=["config", "settings", "file", "format"]
        )
        
        super().__init__(metadata, config)
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute configuration operation."""
        self.status = ToolStatus.RUNNING
        
        try:
            operation_map = {
                'read': self._read_config,
                'write': self._write_config,
                'merge': self._merge_configs,
                'validate': self._validate_config
            }
            
            if operation not in operation_map:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = await operation_map[operation](**kwargs)
            self.status = ToolStatus.COMPLETED
            return {
                'success': True,
                'operation': operation,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.status = ToolStatus.FAILED
            self.logger.error(f"Configuration operation failed: {e}")
            return {
                'success': False,
                'operation': operation,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _read_config(self, file_path: str, 
                          format: Optional[str] = None) -> Dict[str, Any]:
        """Read configuration file."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_directory / path
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Auto-detect format from extension
        if format is None:
            if path.suffix.lower() == '.json':
                format = 'json'
            elif path.suffix.lower() in ['.yml', '.yaml']:
                format = 'yaml'
            elif path.suffix.lower() in ['.ini', '.cfg']:
                format = 'ini'
            elif path.suffix.lower() == '.env':
                format = 'env'
            else:
                format = 'json'  # Default fallback
        
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        if format == 'json':
            config = json.loads(content)
        elif format == 'yaml':
            config = yaml.safe_load(content)
        elif format == 'ini':
            import configparser
            parser = configparser.ConfigParser()
            parser.read_string(content)
            config = {section: dict(parser[section]) for section in parser.sections()}
        elif format == 'env':
            config = {}
            for line in content.strip().split('\n'):
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        else:
            raise ValueError(f"Unsupported configuration format: {format}")
        
        return {
            'path': str(path),
            'format': format,
            'config': config
        }
    
    async def _write_config(self, file_path: str, config: Dict[str, Any],
                           format: Optional[str] = None) -> Dict[str, Any]:
        """Write configuration file."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_directory / path
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect format from extension
        if format is None:
            if path.suffix.lower() == '.json':
                format = 'json'
            elif path.suffix.lower() in ['.yml', '.yaml']:
                format = 'yaml'
            elif path.suffix.lower() in ['.ini', '.cfg']:
                format = 'ini'
            elif path.suffix.lower() == '.env':
                format = 'env'
            else:
                format = 'json'  # Default fallback
        
        if format == 'json':
            content = json.dumps(config, indent=2, ensure_ascii=False)
        elif format == 'yaml':
            content = yaml.dump(config, default_flow_style=False, allow_unicode=True)
        elif format == 'ini':
            import configparser
            parser = configparser.ConfigParser()
            for section, values in config.items():
                parser[section] = values
            import io
            output = io.StringIO()
            parser.write(output)
            content = output.getvalue()
        elif format == 'env':
            lines = []
            for key, value in config.items():
                lines.append(f"{key}={value}")
            content = '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported configuration format: {format}")
        
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(content)
        
        return {
            'path': str(path),
            'format': format,
            'size': path.stat().st_size
        }
    
    async def _merge_configs(self, base_config: str, 
                            override_config: str,
                            output_path: Optional[str] = None) -> Dict[str, Any]:
        """Merge two configuration files."""
        base_data = await self._read_config(base_config)
        override_data = await self._read_config(override_config)
        
        # Deep merge configurations
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_config = deep_merge(base_data['result']['config'], 
                                  override_data['result']['config'])
        
        if output_path:
            await self._write_config(output_path, merged_config, 
                                   format=base_data['result']['format'])
        
        return {
            'base_config': base_config,
            'override_config': override_config,
            'output_path': output_path,
            'merged_config': merged_config
        }
    
    async def _validate_config(self, file_path: str, 
                              schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate configuration file."""
        config_data = await self._read_config(file_path)
        config = config_data['result']['config']
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if schema:
            # Basic schema validation (simplified)
            def validate_against_schema(data: Any, schema_part: Any, path: str = ""):
                if isinstance(schema_part, dict):
                    if 'type' in schema_part:
                        expected_type = schema_part['type']
                        if expected_type == 'object' and not isinstance(data, dict):
                            validation_results['errors'].append(f"{path}: Expected object, got {type(data).__name__}")
                            validation_results['valid'] = False
                        elif expected_type == 'array' and not isinstance(data, list):
                            validation_results['errors'].append(f"{path}: Expected array, got {type(data).__name__}")
                            validation_results['valid'] = False
                        elif expected_type == 'string' and not isinstance(data, str):
                            validation_results['errors'].append(f"{path}: Expected string, got {type(data).__name__}")
                            validation_results['valid'] = False
                        elif expected_type == 'number' and not isinstance(data, (int, float)):
                            validation_results['errors'].append(f"{path}: Expected number, got {type(data).__name__}")
                            validation_results['valid'] = False
                    
                    if 'required' in schema_part and isinstance(data, dict):
                        for required_field in schema_part['required']:
                            if required_field not in data:
                                validation_results['errors'].append(f"{path}: Missing required field '{required_field}'")
                                validation_results['valid'] = False
                    
                    if 'properties' in schema_part and isinstance(data, dict):
                        for prop_name, prop_schema in schema_part['properties'].items():
                            if prop_name in data:
                                validate_against_schema(data[prop_name], prop_schema, f"{path}.{prop_name}")
            
            validate_against_schema(config, schema)
        
        return {
            'config_path': file_path,
            'validation': validation_results,
            'config_summary': {
                'total_keys': len(config) if isinstance(config, dict) else 0,
                'format': config_data['result']['format']
            }
        }

# Import base64 for binary file handling
import base64
