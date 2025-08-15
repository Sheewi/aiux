"""
Action Tokenizer ABI Implementation

Stable binary interface for action tokenization with versioning,
wire format encoding, and backward compatibility guarantees.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import struct
import json
import hashlib
from datetime import datetime
import uuid

from .action_algebra import ActionDefinition, ActionToken, ActionType, ParamType, ParamSchema


class ABIVersion:
    """ABI version management with semantic versioning."""
    
    def __init__(self, major: int, minor: int, patch: int = 0):
        self.major = major
        self.minor = minor
        self.patch = patch
    
    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def is_compatible(self, other: 'ABIVersion') -> bool:
        """Check if this version is backward compatible with other."""
        return (self.major == other.major and 
                self.minor >= other.minor)
    
    def is_breaking_change(self, other: 'ABIVersion') -> bool:
        """Check if upgrading to other would be a breaking change."""
        return other.major > self.major


@dataclass
class ABIEntry:
    """Entry in the ABI table with stable encoding."""
    id: int
    name: str
    action_type: ActionType
    params: Dict[str, ParamSchema]
    caps: Set[str]
    version: ABIVersion
    deprecated: bool = False
    deprecated_since: Optional[ABIVersion] = None
    replacement_id: Optional[int] = None
    
    def to_wire_format(self) -> bytes:
        """Serialize ABI entry to wire format."""
        # Use a simple binary format for stability
        data = {
            'id': self.id,
            'name': self.name,
            'type': int(self.action_type),
            'params': {k: self._serialize_param_schema(v) for k, v in self.params.items()},
            'caps': list(self.caps),
            'version': [self.version.major, self.version.minor, self.version.patch],
            'deprecated': self.deprecated,
            'replacement_id': self.replacement_id
        }
        return json.dumps(data, sort_keys=True).encode('utf-8')
    
    def _serialize_param_schema(self, schema: ParamSchema) -> Dict[str, Any]:
        """Serialize parameter schema to JSON-compatible format."""
        return {
            'type': int(schema.param_type),
            'required': schema.required,
            'default': schema.default,
            'constraints': schema.constraints,
            'enum_values': schema.enum_values
        }
    
    @classmethod
    def from_wire_format(cls, data: bytes) -> 'ABIEntry':
        """Deserialize ABI entry from wire format."""
        obj = json.loads(data.decode('utf-8'))
        
        # Reconstruct parameter schemas
        params = {}
        for name, param_data in obj['params'].items():
            params[name] = ParamSchema(
                param_type=ParamType(param_data['type']),
                required=param_data['required'],
                default=param_data.get('default'),
                constraints=param_data.get('constraints', {}),
                enum_values=param_data.get('enum_values')
            )
        
        version_list = obj['version']
        version = ABIVersion(version_list[0], version_list[1], version_list[2] if len(version_list) > 2 else 0)
        
        return cls(
            id=obj['id'],
            name=obj['name'],
            action_type=ActionType(obj['type']),
            params=params,
            caps=set(obj['caps']),
            version=version,
            deprecated=obj.get('deprecated', False),
            replacement_id=obj.get('replacement_id')
        )


class ActionABI:
    """
    Application Binary Interface for action tokenization.
    
    Manages stable numeric IDs, versioning, and wire format encoding.
    Guarantees backward compatibility and provides conflict resolution.
    """
    
    CURRENT_SCHEMA_VERSION = ABIVersion(1, 0, 0)
    
    def __init__(self):
        self.entries: Dict[int, ABIEntry] = {}
        self.name_to_id: Dict[str, int] = {}
        self.next_id = 1
        self.schema_version = self.CURRENT_SCHEMA_VERSION
        self.created_at = datetime.utcnow()
        self.modified_at = datetime.utcnow()
    
    def register_action(self, name: str, action_type: ActionType, 
                       params: Dict[str, ParamSchema], caps: Set[str],
                       version: ABIVersion = None, action_id: int = None) -> int:
        """
        Register a new action in the ABI.
        
        Args:
            name: Canonical action name (must be uppercase)
            action_type: Type of action
            params: Parameter schemas
            caps: Required capabilities
            version: Action version (defaults to 1.0.0)
            action_id: Explicit ID (for migration/loading)
        
        Returns:
            Assigned action ID
        """
        if not name.isupper():
            raise ValueError(f"Action name must be uppercase: {name}")
        
        if version is None:
            version = ABIVersion(1, 0, 0)
        
        # Check for name conflicts
        if name in self.name_to_id:
            existing_id = self.name_to_id[name]
            existing = self.entries[existing_id]
            if not existing.deprecated:
                raise ValueError(f"Action name {name} already registered with ID {existing_id}")
        
        # Assign ID
        if action_id is None:
            action_id = self._allocate_id()
        else:
            if action_id in self.entries:
                existing = self.entries[action_id]
                if existing.name != name:
                    raise ValueError(f"ID {action_id} already assigned to {existing.name}")
                # Allow re-registration with same ID/name for updates
            self.next_id = max(self.next_id, action_id + 1)
        
        # Create entry
        entry = ABIEntry(
            id=action_id,
            name=name,
            action_type=action_type,
            params=params,
            caps=caps,
            version=version
        )
        
        self.entries[action_id] = entry
        self.name_to_id[name] = action_id
        self.modified_at = datetime.utcnow()
        
        return action_id
    
    def deprecate_action(self, identifier: Union[int, str], 
                        replacement_id: Optional[int] = None) -> None:
        """Deprecate an action (mark as deprecated but keep in ABI)."""
        entry = self.get_entry(identifier)
        if not entry:
            raise ValueError(f"Action not found: {identifier}")
        
        entry.deprecated = True
        entry.deprecated_since = self.schema_version
        entry.replacement_id = replacement_id
        self.modified_at = datetime.utcnow()
    
    def get_entry(self, identifier: Union[int, str]) -> Optional[ABIEntry]:
        """Get ABI entry by ID or name."""
        if isinstance(identifier, int):
            return self.entries.get(identifier)
        else:
            action_id = self.name_to_id.get(identifier)
            return self.entries.get(action_id) if action_id else None
    
    def get_active_entries(self) -> List[ABIEntry]:
        """Get all non-deprecated entries."""
        return [entry for entry in self.entries.values() if not entry.deprecated]
    
    def create_token(self, identifier: Union[int, str], args: Dict[str, Any], 
                    **meta) -> ActionToken:
        """Create an action token using this ABI."""
        entry = self.get_entry(identifier)
        if not entry:
            raise ValueError(f"Unknown action: {identifier}")
        
        if entry.deprecated:
            if entry.replacement_id:
                replacement = self.entries.get(entry.replacement_id)
                if replacement:
                    print(f"Warning: {entry.name} is deprecated, use {replacement.name} instead")
            else:
                print(f"Warning: {entry.name} is deprecated")
        
        # Create action definition for validation
        definition = ActionDefinition(
            id=entry.id,
            name=entry.name,
            action_type=entry.action_type,
            params=entry.params,
            caps=frozenset(entry.caps),
            version=(entry.version.major, entry.version.minor)
        )
        
        return definition.create_token(args, **meta)
    
    def validate_token(self, token: ActionToken) -> List[str]:
        """Validate a token against the current ABI."""
        errors = []
        
        entry = self.get_entry(token.id)
        if not entry:
            errors.append(f"Unknown action ID: {token.id}")
            return errors
        
        if entry.name != token.name:
            errors.append(f"Name mismatch: expected {entry.name}, got {token.name}")
        
        if entry.action_type != token.action_type:
            errors.append(f"Type mismatch: expected {entry.action_type}, got {token.action_type}")
        
        # Validate arguments
        for param_name, schema in entry.params.items():
            if param_name in token.args:
                if not schema.validate(token.args[param_name]):
                    errors.append(f"Invalid value for {param_name}")
            elif schema.required and schema.default is None:
                errors.append(f"Missing required parameter: {param_name}")
        
        # Check for unknown arguments
        unknown_args = set(token.args.keys()) - set(entry.params.keys())
        if unknown_args:
            errors.append(f"Unknown arguments: {unknown_args}")
        
        return errors
    
    def export_abi_table(self) -> str:
        """Export the complete ABI table as JSON."""
        data = {
            'schema_version': str(self.schema_version),
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'entries': []
        }
        
        for entry in sorted(self.entries.values(), key=lambda x: x.id):
            entry_data = {
                'id': entry.id,
                'name': entry.name,
                'type': entry.action_type.name,
                'params': {},
                'caps': sorted(entry.caps),
                'version': str(entry.version),
                'deprecated': entry.deprecated
            }
            
            # Format parameters in human-readable form
            for param_name, schema in entry.params.items():
                param_str = schema.param_type.name.lower()
                if not schema.required:
                    param_str += "?"
                if schema.default is not None:
                    param_str += f"={schema.default}"
                if schema.enum_values:
                    param_str += f"({','.join(schema.enum_values)})"
                entry_data['params'][param_name] = param_str
            
            if entry.replacement_id:
                entry_data['replacement_id'] = entry.replacement_id
            
            data['entries'].append(entry_data)
        
        return json.dumps(data, indent=2)
    
    def load_abi_table(self, json_data: str) -> None:
        """Load ABI table from JSON export."""
        data = json.loads(json_data)
        
        # Check version compatibility
        loaded_version = ABIVersion(*map(int, data['schema_version'].split('.')))
        if not self.schema_version.is_compatible(loaded_version):
            raise ValueError(f"Incompatible schema version: {loaded_version}")
        
        # Clear current state
        self.entries.clear()
        self.name_to_id.clear()
        self.next_id = 1
        
        # Load entries
        for entry_data in data['entries']:
            # Parse parameter schemas from string format
            params = {}
            for param_name, param_str in entry_data['params'].items():
                # Simple parsing - could be more sophisticated
                param_type = ParamType[param_str.split('?')[0].split('=')[0].split('(')[0].upper()]
                required = '?' not in param_str
                default = None
                enum_values = None
                
                if '=' in param_str:
                    default_part = param_str.split('=')[1].split('(')[0]
                    try:
                        default = json.loads(default_part)
                    except:
                        default = default_part
                
                if '(' in param_str and ')' in param_str:
                    enum_part = param_str.split('(')[1].split(')')[0]
                    enum_values = enum_part.split(',')
                
                params[param_name] = ParamSchema(
                    param_type=param_type,
                    required=required,
                    default=default,
                    enum_values=enum_values
                )
            
            # Create entry
            version = ABIVersion(*map(int, entry_data['version'].split('.')))
            entry = ABIEntry(
                id=entry_data['id'],
                name=entry_data['name'],
                action_type=ActionType[entry_data['type']],
                params=params,
                caps=set(entry_data['caps']),
                version=version,
                deprecated=entry_data.get('deprecated', False),
                replacement_id=entry_data.get('replacement_id')
            )
            
            self.entries[entry.id] = entry
            self.name_to_id[entry.name] = entry.id
            self.next_id = max(self.next_id, entry.id + 1)
    
    def _allocate_id(self) -> int:
        """Allocate the next available ID."""
        action_id = self.next_id
        self.next_id += 1
        return action_id
    
    def compute_abi_hash(self) -> str:
        """Compute stable hash of the ABI for versioning."""
        # Sort entries by ID for stability
        sorted_entries = sorted(self.entries.values(), key=lambda x: x.id)
        content = []
        
        for entry in sorted_entries:
            if not entry.deprecated:  # Only include active entries
                content.append((entry.id, entry.name, int(entry.action_type),
                              tuple(sorted(entry.params.keys())),
                              tuple(sorted(entry.caps))))
        
        return hashlib.sha256(str(content).encode()).hexdigest()[:16]


# Default ABI with core actions
def create_default_abi() -> ActionABI:
    """Create the default ABI with standard actions."""
    abi = ActionABI()
    
    # Core actions from the specification
    actions = [
        (1, "CLICK", ActionType.ACTION, {
            "target": ParamSchema(ParamType.SELECTOR, required=True),
            "button": ParamSchema(ParamType.ENUM, required=False, default="primary", 
                                enum_values=["primary", "secondary"]),
            "count": ParamSchema(ParamType.INT, required=False, default=1)
        }, {"dom.query"}),
        
        (2, "TYPE", ActionType.ACTION, {
            "target": ParamSchema(ParamType.SELECTOR, required=True),
            "text": ParamSchema(ParamType.STRING, required=True),
            "enter": ParamSchema(ParamType.BOOL, required=False, default=False)
        }, {"dom.query", "input"}),
        
        (3, "WAIT", ActionType.CONTROL, {
            "duration": ParamSchema(ParamType.DURATION, required=True)
        }, {"timing"}),
        
        (4, "NAVIGATE", ActionType.ACTION, {
            "url": ParamSchema(ParamType.URL, required=True)
        }, {"net.fetch"}),
        
        (5, "SET", ActionType.ACTION, {
            "key": ParamSchema(ParamType.KVKEY, required=True),
            "value": ParamSchema(ParamType.KVVALUE, required=True),
            "scope": ParamSchema(ParamType.ENUM, required=False, default="session",
                               enum_values=["session", "global"])
        }, {"storage"}),
        
        (6, "ASSERT", ActionType.CONTROL, {
            "predicate": ParamSchema(ParamType.PREDICATE, required=True),
            "timeout": ParamSchema(ParamType.DURATION, required=False, default=0)
        }, {"dom.query"}),
        
        (7, "CAPTURE_IMG", ActionType.SENSOR, {
            "region": ParamSchema(ParamType.BOX, required=False),
            "label": ParamSchema(ParamType.STRING, required=False)
        }, {"camera", "screen"}),
        
        (8, "LOOP", ActionType.CONTROL, {
            "times": ParamSchema(ParamType.INT, required=True)
        }, {"control"}),
    ]
    
    for action_id, name, action_type, params, caps in actions:
        abi.register_action(name, action_type, params, caps, 
                           action_id=action_id)
    
    return abi
