"""
Action Algebra Implementation

Formal mathematical algebra for action tokens with composition operations,
type safety, and constraint validation.
"""

from typing import Dict, List, Any, Optional, Union, Set, Tuple, FrozenSet
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import hashlib
from datetime import datetime
from enum import Enum, IntEnum

# Optional dependency for binary serialization
try:
    import cbor2
    HAS_CBOR = True
except ImportError:
    HAS_CBOR = False


class ActionType(IntEnum):
    """Action types with stable numeric encoding."""
    ACTION = 1    # Direct world-changing operations
    CONTROL = 2   # Flow control and orchestration
    SENSOR = 3    # Data collection and observation
    EVENT = 4     # Event handling and reactive operations


class ParamType(IntEnum):
    """Parameter types with stable encoding."""
    STRING = 1
    INT = 2
    FLOAT = 3
    BOOL = 4
    SELECTOR = 5
    DURATION = 6
    URL = 7
    KVKEY = 8
    KVVALUE = 9
    PREDICATE = 10
    BOX = 11
    ENUM = 12
    LIST = 13
    DICT = 14


@dataclass(frozen=True)
class ParamSchema:
    """Type schema for action parameters."""
    param_type: ParamType
    required: bool = True
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    enum_values: Optional[List[str]] = None
    
    def validate(self, value: Any) -> bool:
        """Validate a value against this schema."""
        if value is None:
            return not self.required
        
        # Type validation
        if self.param_type == ParamType.STRING:
            if not isinstance(value, str):
                return False
        elif self.param_type == ParamType.INT:
            if not isinstance(value, int):
                return False
        elif self.param_type == ParamType.FLOAT:
            if not isinstance(value, (int, float)):
                return False
        elif self.param_type == ParamType.BOOL:
            if not isinstance(value, bool):
                return False
        elif self.param_type == ParamType.ENUM:
            if self.enum_values and value not in self.enum_values:
                return False
        
        # Constraint validation
        if 'min' in self.constraints and value < self.constraints['min']:
            return False
        if 'max' in self.constraints and value > self.constraints['max']:
            return False
        if 'pattern' in self.constraints:
            import re
            if not re.match(self.constraints['pattern'], str(value)):
                return False
        
        return True


@dataclass(frozen=True)
class ActionToken:
    """
    Core action token with stable ABI.
    
    Immutable token representing a single action in the algebra.
    """
    id: int                                    # Stable 32-bit ABI identifier
    name: str                                  # Canonical action name
    action_type: ActionType                    # Type classification
    args: Dict[str, Any]                       # Typed arguments
    caps: FrozenSet[str]                       # Required capabilities
    meta: Dict[str, Any] = field(default_factory=dict)  # Metadata/provenance
    
    def __post_init__(self):
        """Validate token construction."""
        if not (0 <= self.id <= 0xFFFFFFFF):
            raise ValueError(f"Action ID {self.id} outside 32-bit range")
        
        if not self.name.isupper():
            raise ValueError(f"Action name {self.name} must be uppercase")
    
    def to_wire_format(self) -> bytes:
        """Serialize to binary wire format (CBOR)."""
        data = {
            'id': self.id,
            'name': self.name,
            'type': int(self.action_type),
            'args': self.args,
            'caps': list(self.caps),
            'meta': self.meta
        }
        return cbor2.dumps(data)
    
    @classmethod
    def from_wire_format(cls, data: bytes) -> 'ActionToken':
        """Deserialize from binary wire format."""
        obj = cbor2.loads(data)
        return cls(
            id=obj['id'],
            name=obj['name'],
            action_type=ActionType(obj['type']),
            args=obj['args'],
            caps=frozenset(obj['caps']),
            meta=obj.get('meta', {})
        )
    
    def to_json(self) -> str:
        """Serialize to human-readable JSON."""
        data = {
            'id': self.id,
            'name': self.name,
            'type': self.action_type.name,
            'args': self.args,
            'caps': list(self.caps),
            'meta': self.meta
        }
        return json.dumps(data, indent=2)
    
    def hash_signature(self) -> str:
        """Compute stable hash signature for deduplication."""
        # Hash based on structural content, not metadata
        content = (self.id, self.name, self.action_type, 
                  tuple(sorted(self.args.items())), tuple(sorted(self.caps)))
        return hashlib.sha256(str(content).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ActionDefinition:
    """
    Formal definition of an action in the algebra.
    
    Defines the schema, capabilities, and behavioral contract for an action type.
    """
    id: int
    name: str
    action_type: ActionType
    params: Dict[str, ParamSchema]
    caps: FrozenSet[str]
    version: Tuple[int, int]  # (major, minor)
    description: str = ""
    effects: Dict[str, str] = field(default_factory=dict)  # State effects
    dependencies: FrozenSet[int] = field(default_factory=frozenset)  # Action IDs this depends on
    
    def create_token(self, args: Dict[str, Any], **meta) -> ActionToken:
        """Create a token instance from this definition."""
        # Validate all arguments
        validated_args = {}
        
        for param_name, schema in self.params.items():
            if param_name in args:
                value = args[param_name]
                if not schema.validate(value):
                    raise ValueError(f"Invalid value {value} for parameter {param_name}")
                validated_args[param_name] = value
            elif schema.required:
                if schema.default is not None:
                    validated_args[param_name] = schema.default
                else:
                    raise ValueError(f"Required parameter {param_name} missing")
            elif schema.default is not None:
                validated_args[param_name] = schema.default
        
        # Check for unexpected arguments
        unexpected = set(args.keys()) - set(self.params.keys())
        if unexpected:
            raise ValueError(f"Unexpected arguments: {unexpected}")
        
        return ActionToken(
            id=self.id,
            name=self.name,
            action_type=self.action_type,
            args=validated_args,
            caps=self.caps,
            meta=meta
        )


class ActionAlgebra:
    """
    The formal action algebra with composition operations.
    
    Implements the monoid structure ⟨A,;,ε⟩ with sequential composition,
    guarded execution, and scoped operations.
    """
    
    def __init__(self):
        self.definitions: Dict[int, ActionDefinition] = {}
        self.name_to_id: Dict[str, int] = {}
        self.dependency_graph: Dict[int, Set[int]] = {}
    
    def register_action(self, definition: ActionDefinition) -> None:
        """Register an action definition in the algebra."""
        if definition.id in self.definitions:
            existing = self.definitions[definition.id]
            if existing.name != definition.name:
                raise ValueError(f"ID collision: {definition.id} used for both {existing.name} and {definition.name}")
        
        if definition.name in self.name_to_id:
            existing_id = self.name_to_id[definition.name]
            if existing_id != definition.id:
                raise ValueError(f"Name collision: {definition.name} used for both {existing_id} and {definition.id}")
        
        self.definitions[definition.id] = definition
        self.name_to_id[definition.name] = definition.id
        self.dependency_graph[definition.id] = definition.dependencies
    
    def get_definition(self, identifier: Union[int, str]) -> Optional[ActionDefinition]:
        """Get action definition by ID or name."""
        if isinstance(identifier, int):
            return self.definitions.get(identifier)
        else:
            action_id = self.name_to_id.get(identifier)
            return self.definitions.get(action_id) if action_id else None
    
    def sequence(self, *actions: ActionToken) -> 'ActionSequence':
        """Create sequential composition of actions."""
        return ActionSequence(list(actions))
    
    def guard(self, condition: str, then_action: ActionToken, 
              else_action: Optional[ActionToken] = None) -> 'GuardedAction':
        """Create conditional action execution."""
        return GuardedAction(condition, then_action, else_action)
    
    def scope(self, context: Dict[str, Any], action: ActionToken) -> 'ScopedAction':
        """Create scoped action execution."""
        return ScopedAction(context, action)
    
    def validate_sequence(self, sequence: List[ActionToken]) -> List[str]:
        """Validate a sequence of actions for dependency violations."""
        errors = []
        satisfied_effects = set()
        
        for i, token in enumerate(sequence):
            definition = self.get_definition(token.id)
            if not definition:
                errors.append(f"Unknown action at position {i}: {token.name}")
                continue
            
            # Check dependencies
            for dep_id in definition.dependencies:
                dep_def = self.get_definition(dep_id)
                if not dep_def:
                    errors.append(f"Unknown dependency {dep_id} for {token.name}")
                    continue
                
                # Check if dependency's effects are satisfied
                dep_effects = set(dep_def.effects.keys())
                if not dep_effects.issubset(satisfied_effects):
                    missing = dep_effects - satisfied_effects
                    errors.append(f"Unsatisfied dependencies for {token.name}: {missing}")
            
            # Add this action's effects
            satisfied_effects.update(definition.effects.keys())
        
        return errors
    
    def topological_sort(self, action_ids: Set[int]) -> List[int]:
        """Sort actions by dependency order."""
        result = []
        visited = set()
        temp_visited = set()
        
        def visit(action_id):
            if action_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving action {action_id}")
            if action_id in visited:
                return
            
            temp_visited.add(action_id)
            for dep_id in self.dependency_graph.get(action_id, set()):
                if dep_id in action_ids:  # Only consider dependencies in our set
                    visit(dep_id)
            
            temp_visited.remove(action_id)
            visited.add(action_id)
            result.append(action_id)
        
        for action_id in action_ids:
            if action_id not in visited:
                visit(action_id)
        
        return result


@dataclass
class ActionSequence:
    """Sequential composition of actions (A ; B ; C)."""
    actions: List[ActionToken]
    
    def __iter__(self):
        return iter(self.actions)
    
    def __len__(self):
        return len(self.actions)
    
    def append(self, action: ActionToken):
        """Add action to sequence."""
        self.actions.append(action)
    
    def to_wire_format(self) -> bytes:
        """Serialize sequence to wire format."""
        return cbor2.dumps([action.to_wire_format() for action in self.actions])


@dataclass
class GuardedAction:
    """Conditional action execution IF(condition) { then_action } [ELSE { else_action }]."""
    condition: str
    then_action: ActionToken
    else_action: Optional[ActionToken] = None


@dataclass
class ScopedAction:
    """Scoped action execution WITH(context) { action }."""
    context: Dict[str, Any]
    action: ActionToken


# Identity element (monoid)
NOOP = ActionToken(
    id=0,
    name="NOOP",
    action_type=ActionType.CONTROL,
    args={},
    caps=frozenset(),
    meta={"description": "Identity element - no operation"}
)
