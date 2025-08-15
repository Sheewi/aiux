"""
Action Grammar System

Defines formal grammar for action tokens with context-free productions,
semantic validation, and compositional parsing rules.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import re
from collections import defaultdict

from .action_algebra import ActionToken, ActionSequence, ActionAlgebra
from .action_abi import ActionABI


class TokenType(Enum):
    """Token types in the action grammar."""
    ACTION = "action"
    ARGUMENT = "argument"
    SELECTOR = "selector"
    VALUE = "value"
    OPERATOR = "operator"
    DELIMITER = "delimiter"
    MODIFIER = "modifier"
    CONDITION = "condition"


@dataclass
class ParsedToken:
    """A token parsed from input with grammatical information."""
    value: str
    token_type: TokenType
    position: Tuple[int, int]  # Start, end positions
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Production:
    """Grammar production rule."""
    lhs: str  # Left-hand side (non-terminal)
    rhs: List[str]  # Right-hand side (terminals and non-terminals)
    semantic_action: Optional[str] = None
    priority: float = 1.0
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseNode:
    """Node in the parse tree."""
    symbol: str
    value: Optional[str] = None
    children: List['ParseNode'] = field(default_factory=list)
    token: Optional[ParsedToken] = None
    semantic_value: Any = None
    position: Tuple[int, int] = (0, 0)


class SemanticValidator:
    """Validates semantic constraints on parse trees."""
    
    def __init__(self, abi: ActionABI):
        self.abi = abi
        self.validators = {
            'action_exists': self._validate_action_exists,
            'argument_type': self._validate_argument_type,
            'selector_valid': self._validate_selector,
            'value_format': self._validate_value_format,
            'capability_required': self._validate_capability
        }
    
    def validate(self, node: ParseNode, constraint: str, params: Dict[str, Any]) -> bool:
        """Validate a semantic constraint."""
        if constraint in self.validators:
            return self.validators[constraint](node, params)
        return True
    
    def _validate_action_exists(self, node: ParseNode, params: Dict[str, Any]) -> bool:
        """Validate that action exists in ABI."""
        if node.symbol == 'ACTION' and node.value:
            return self.abi.has_action(node.value)
        return True
    
    def _validate_argument_type(self, node: ParseNode, params: Dict[str, Any]) -> bool:
        """Validate argument type compatibility."""
        expected_type = params.get('type')
        if not expected_type:
            return True
        
        # Simple type checking (could be more sophisticated)
        if expected_type == 'string':
            return isinstance(node.semantic_value, str)
        elif expected_type == 'number':
            return isinstance(node.semantic_value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(node.semantic_value, bool)
        
        return True
    
    def _validate_selector(self, node: ParseNode, params: Dict[str, Any]) -> bool:
        """Validate CSS/XPath selector syntax."""
        if node.symbol == 'SELECTOR' and node.value:
            # Basic selector validation
            selector = node.value
            
            # CSS selector patterns
            css_patterns = [
                r'^#[\w-]+$',  # ID
                r'^\.[\w-]+$',  # Class
                r'^\w+$',  # Tag
                r'^\[[\w-]+.*\]$',  # Attribute
            ]
            
            # XPath patterns
            xpath_patterns = [
                r'^//.*',  # XPath
                r'^/.*',   # Absolute XPath
            ]
            
            all_patterns = css_patterns + xpath_patterns
            return any(re.match(pattern, selector) for pattern in all_patterns)
        
        return True
    
    def _validate_value_format(self, node: ParseNode, params: Dict[str, Any]) -> bool:
        """Validate value format (URL, email, etc.)."""
        format_type = params.get('format')
        if not format_type or not node.value:
            return True
        
        if format_type == 'url':
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            return bool(re.match(url_pattern, node.value))
        elif format_type == 'email':
            email_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
            return bool(re.match(email_pattern, node.value))
        
        return True
    
    def _validate_capability(self, node: ParseNode, params: Dict[str, Any]) -> bool:
        """Validate capability requirements."""
        # This would check against available capabilities in context
        return True


class ActionGrammar:
    """
    Context-free grammar for action language.
    
    Grammar productions for mapping natural language to action tokens.
    """
    
    def __init__(self, abi: ActionABI = None):
        self.abi = abi
        self.productions = []
        self.semantic_validator = SemanticValidator(abi) if abi else None
        self.start_symbol = "S"
        
        # Initialize default grammar
        self._load_default_grammar()
    
    def _load_default_grammar(self):
        """Load default grammar productions."""
        
        # Start rule
        self.add_production("S", ["ACTION_SEQUENCE"])
        
        # Action sequences
        self.add_production("ACTION_SEQUENCE", ["ACTION"])
        self.add_production("ACTION_SEQUENCE", ["ACTION", "CONJUNCTION", "ACTION_SEQUENCE"])
        
        # Actions
        self.add_production("ACTION", ["SIMPLE_ACTION"])
        self.add_production("ACTION", ["MODIFIED_ACTION"])
        self.add_production("ACTION", ["CONDITIONAL_ACTION"])
        
        # Simple actions
        self.add_production("SIMPLE_ACTION", ["CLICK_ACTION"])
        self.add_production("SIMPLE_ACTION", ["TYPE_ACTION"])
        self.add_production("SIMPLE_ACTION", ["NAVIGATE_ACTION"])
        self.add_production("SIMPLE_ACTION", ["WAIT_ACTION"])
        self.add_production("SIMPLE_ACTION", ["CAPTURE_ACTION"])
        
        # Specific action patterns
        self.add_production("CLICK_ACTION", ["CLICK", "TARGET"], 
                          semantic_action="create_click_token")
        self.add_production("CLICK_ACTION", ["CLICK", "ON", "TARGET"],
                          semantic_action="create_click_token")
        
        self.add_production("TYPE_ACTION", ["TYPE", "TEXT", "IN", "TARGET"],
                          semantic_action="create_type_token")
        self.add_production("TYPE_ACTION", ["TYPE", "TEXT", "INTO", "TARGET"],
                          semantic_action="create_type_token")
        
        self.add_production("NAVIGATE_ACTION", ["NAVIGATE", "TO", "URL"],
                          semantic_action="create_navigate_token")
        self.add_production("NAVIGATE_ACTION", ["GO", "TO", "URL"],
                          semantic_action="create_navigate_token")
        
        self.add_production("WAIT_ACTION", ["WAIT", "DURATION"],
                          semantic_action="create_wait_token")
        
        self.add_production("CAPTURE_ACTION", ["CAPTURE", "SCREENSHOT"],
                          semantic_action="create_capture_token")
        self.add_production("CAPTURE_ACTION", ["TAKE", "SCREENSHOT"],
                          semantic_action="create_capture_token")
        
        # Modified actions
        self.add_production("MODIFIED_ACTION", ["MODIFIER", "ACTION"])
        
        # Conditional actions
        self.add_production("CONDITIONAL_ACTION", ["IF", "CONDITION", "THEN", "ACTION"])
        
        # Terminals and basic structures
        self.add_production("TARGET", ["SELECTOR"])
        self.add_production("TARGET", ["QUOTED_STRING"])
        
        self.add_production("TEXT", ["QUOTED_STRING"])
        self.add_production("URL", ["QUOTED_STRING"])
        self.add_production("URL", ["URL_LITERAL"])
        
        self.add_production("DURATION", ["NUMBER", "TIME_UNIT"])
        self.add_production("DURATION", ["NUMBER"])
        
        self.add_production("CONDITION", ["ELEMENT_EXISTS"])
        self.add_production("CONDITION", ["ELEMENT_VISIBLE"])
        
        # Lexical rules (terminals)
        self._add_terminal_patterns()
    
    def _add_terminal_patterns(self):
        """Add terminal token patterns."""
        self.terminal_patterns = {
            'CLICK': [r'click', r'tap', r'press'],
            'TYPE': [r'type', r'enter', r'input'],
            'NAVIGATE': [r'navigate', r'visit', r'open'],
            'GO': [r'go'],
            'WAIT': [r'wait', r'pause', r'delay'],
            'CAPTURE': [r'capture', r'take', r'grab'],
            'TAKE': [r'take'],
            
            'ON': [r'on'],
            'IN': [r'in', r'inside'],
            'INTO': [r'into'],
            'TO': [r'to'],
            'IF': [r'if'],
            'THEN': [r'then'],
            
            'SCREENSHOT': [r'screenshot', r'image', r'picture'],
            
            'CONJUNCTION': [r'and', r'then', r','],
            
            'QUOTED_STRING': [r'"([^"]*)"', r"'([^']*)'"],
            'SELECTOR': [
                r'#[\w-]+',  # ID selector
                r'\.[\w-]+',  # Class selector
                r'//[^\s]+',  # XPath
                r'\[[\w-]+.*?\]',  # Attribute selector
            ],
            'URL_LITERAL': [r'https?://[^\s]+'],
            'NUMBER': [r'\d+(?:\.\d+)?'],
            'TIME_UNIT': [r'second[s]?', r'sec[s]?', r'minute[s]?', r'min[s]?', r'hour[s]?', r'hr[s]?'],
            
            'MODIFIER': [r'quickly', r'slowly', r'carefully'],
            'ELEMENT_EXISTS': [r'element\s+exists'],
            'ELEMENT_VISIBLE': [r'element\s+(?:is\s+)?visible'],
        }
    
    def add_production(self, lhs: str, rhs: List[str], 
                      semantic_action: str = None, priority: float = 1.0,
                      constraints: Dict[str, Any] = None):
        """Add a production rule to the grammar."""
        production = Production(
            lhs=lhs,
            rhs=rhs,
            semantic_action=semantic_action,
            priority=priority,
            constraints=constraints or {}
        )
        self.productions.append(production)
    
    def get_productions_for(self, symbol: str) -> List[Production]:
        """Get all productions with the given symbol on the LHS."""
        return [p for p in self.productions if p.lhs == symbol]
    
    def is_terminal(self, symbol: str) -> bool:
        """Check if a symbol is terminal."""
        return symbol in self.terminal_patterns
    
    def match_terminal(self, symbol: str, text: str, position: int) -> Optional[ParsedToken]:
        """Match a terminal symbol against text at position."""
        if symbol not in self.terminal_patterns:
            return None
        
        patterns = self.terminal_patterns[symbol]
        remaining_text = text[position:]
        
        for pattern in patterns:
            match = re.match(pattern, remaining_text, re.IGNORECASE)
            if match:
                matched_text = match.group(0)
                # Extract captured group if it exists (for quoted strings)
                value = match.group(1) if match.groups() else matched_text
                
                return ParsedToken(
                    value=value,
                    token_type=self._get_token_type(symbol),
                    position=(position, position + len(matched_text)),
                    confidence=1.0,
                    metadata={'pattern': pattern, 'full_match': matched_text}
                )
        
        return None
    
    def _get_token_type(self, symbol: str) -> TokenType:
        """Map grammar symbol to token type."""
        type_mapping = {
            'CLICK': TokenType.ACTION,
            'TYPE': TokenType.ACTION,
            'NAVIGATE': TokenType.ACTION,
            'WAIT': TokenType.ACTION,
            'CAPTURE': TokenType.ACTION,
            
            'QUOTED_STRING': TokenType.VALUE,
            'SELECTOR': TokenType.SELECTOR,
            'URL_LITERAL': TokenType.VALUE,
            'NUMBER': TokenType.VALUE,
            
            'CONJUNCTION': TokenType.OPERATOR,
            'MODIFIER': TokenType.MODIFIER,
            'CONDITION': TokenType.CONDITION,
        }
        
        return type_mapping.get(symbol, TokenType.VALUE)


class ChartParser:
    """
    Chart parser for action grammar using Earley algorithm.
    
    Handles ambiguous grammars and produces all valid parses.
    """
    
    def __init__(self, grammar: ActionGrammar):
        self.grammar = grammar
        self.chart = []
        self.semantic_actions = self._load_semantic_actions()
    
    def parse(self, tokens: List[str]) -> List[ParseNode]:
        """Parse token sequence and return parse trees."""
        self.chart = [[] for _ in range(len(tokens) + 1)]
        
        # Initialize chart with start symbol
        start_item = ChartItem(
            production=Production(lhs="S'", rhs=[self.grammar.start_symbol]),
            dot_position=0,
            start=0,
            end=0
        )
        self.chart[0].append(start_item)
        
        # Parse each position
        for i in range(len(tokens) + 1):
            self._process_chart_position(i, tokens)
        
        # Extract parse trees
        return self._extract_parse_trees(tokens)
    
    def _process_chart_position(self, position: int, tokens: List[str]):
        """Process all items at a chart position."""
        agenda = list(self.chart[position])
        processed = set()
        
        while agenda:
            item = agenda.pop(0)
            if item in processed:
                continue
            processed.add(item)
            
            if item.is_complete():
                # Complete item - process all items that expect this non-terminal
                self._complete(item, position, agenda)
            elif item.next_symbol_is_terminal():
                # Terminal expected - try to scan
                if position < len(tokens):
                    self._scan(item, tokens[position], position, agenda)
            else:
                # Non-terminal expected - predict
                self._predict(item, position, agenda)
    
    def _predict(self, item: ChartItem, position: int, agenda: List['ChartItem']):
        """Predict new items for non-terminal."""
        next_symbol = item.next_symbol()
        if not next_symbol:
            return
        
        for production in self.grammar.get_productions_for(next_symbol):
            new_item = ChartItem(
                production=production,
                dot_position=0,
                start=position,
                end=position
            )
            
            if new_item not in self.chart[position]:
                self.chart[position].append(new_item)
                agenda.append(new_item)
    
    def _scan(self, item: ChartItem, token: str, position: int, agenda: List['ChartItem']):
        """Scan terminal symbol."""
        expected_symbol = item.next_symbol()
        if not expected_symbol:
            return
        
        # Try to match terminal
        parsed_token = self.grammar.match_terminal(expected_symbol, token, 0)
        if parsed_token:
            new_item = ChartItem(
                production=item.production,
                dot_position=item.dot_position + 1,
                start=item.start,
                end=position + 1,
                parsed_tokens=item.parsed_tokens + [parsed_token]
            )
            
            if new_item not in self.chart[position + 1]:
                self.chart[position + 1].append(new_item)
    
    def _complete(self, completed_item: ChartItem, position: int, agenda: List['ChartItem']):
        """Complete item by advancing dot in items that expect this non-terminal."""
        completed_symbol = completed_item.production.lhs
        
        for item in self.chart[completed_item.start]:
            if item.next_symbol() == completed_symbol:
                new_item = ChartItem(
                    production=item.production,
                    dot_position=item.dot_position + 1,
                    start=item.start,
                    end=position,
                    parsed_tokens=item.parsed_tokens,
                    children=item.children + [completed_item]
                )
                
                if new_item not in self.chart[position]:
                    self.chart[position].append(new_item)
                    agenda.append(new_item)
    
    def _extract_parse_trees(self, tokens: List[str]) -> List[ParseNode]:
        """Extract parse trees from completed chart."""
        parse_trees = []
        final_position = len(tokens)
        
        for item in self.chart[final_position]:
            if (item.production.lhs == "S'" and 
                item.is_complete() and 
                item.start == 0):
                
                # Build parse tree from chart item
                tree = self._build_parse_tree(item)
                if tree:
                    parse_trees.append(tree)
        
        return parse_trees
    
    def _build_parse_tree(self, item: ChartItem) -> Optional[ParseNode]:
        """Build parse tree from chart item."""
        if not item.children:
            # Terminal leaf
            if item.parsed_tokens:
                token = item.parsed_tokens[-1]
                return ParseNode(
                    symbol=item.production.rhs[item.dot_position - 1],
                    value=token.value,
                    token=token,
                    position=token.position
                )
            return None
        
        # Non-terminal node
        node = ParseNode(
            symbol=item.production.lhs,
            position=(item.start, item.end)
        )
        
        # Build children
        for child_item in item.children:
            child_node = self._build_parse_tree(child_item)
            if child_node:
                node.children.append(child_node)
        
        # Apply semantic action
        if item.production.semantic_action:
            node.semantic_value = self._apply_semantic_action(
                item.production.semantic_action, node
            )
        
        return node
    
    def _load_semantic_actions(self) -> Dict[str, Callable]:
        """Load semantic action functions."""
        return {
            'create_click_token': self._create_click_token,
            'create_type_token': self._create_type_token,
            'create_navigate_token': self._create_navigate_token,
            'create_wait_token': self._create_wait_token,
            'create_capture_token': self._create_capture_token,
        }
    
    def _apply_semantic_action(self, action_name: str, node: ParseNode) -> Any:
        """Apply semantic action to parse node."""
        if action_name in self.semantic_actions:
            return self.semantic_actions[action_name](node)
        return None
    
    def _create_click_token(self, node: ParseNode) -> ActionToken:
        """Create CLICK action token from parse tree."""
        target = self._extract_target(node)
        return ActionToken(
            id=self._generate_token_id(),
            name="CLICK",
            type="action",
            args={"target": target},
            caps={"ui", "automation"},
            meta={"source": "grammar_parser"}
        )
    
    def _create_type_token(self, node: ParseNode) -> ActionToken:
        """Create TYPE action token from parse tree."""
        text = self._extract_text(node)
        target = self._extract_target(node)
        return ActionToken(
            id=self._generate_token_id(),
            name="TYPE",
            type="action",
            args={"text": text, "target": target},
            caps={"ui", "automation"},
            meta={"source": "grammar_parser"}
        )
    
    def _create_navigate_token(self, node: ParseNode) -> ActionToken:
        """Create NAVIGATE action token from parse tree."""
        url = self._extract_url(node)
        return ActionToken(
            id=self._generate_token_id(),
            name="NAVIGATE",
            type="action",
            args={"url": url},
            caps={"web", "navigation"},
            meta={"source": "grammar_parser"}
        )
    
    def _create_wait_token(self, node: ParseNode) -> ActionToken:
        """Create WAIT action token from parse tree."""
        duration = self._extract_duration(node)
        return ActionToken(
            id=self._generate_token_id(),
            name="WAIT",
            type="action",
            args={"duration": duration},
            caps={"timing"},
            meta={"source": "grammar_parser"}
        )
    
    def _create_capture_token(self, node: ParseNode) -> ActionToken:
        """Create CAPTURE_IMG action token from parse tree."""
        return ActionToken(
            id=self._generate_token_id(),
            name="CAPTURE_IMG",
            type="action",
            args={},
            caps={"screenshot", "imaging"},
            meta={"source": "grammar_parser"}
        )
    
    def _extract_target(self, node: ParseNode) -> str:
        """Extract target from parse tree."""
        for child in node.children:
            if child.symbol == "TARGET":
                return self._extract_value_from_node(child)
        return ""
    
    def _extract_text(self, node: ParseNode) -> str:
        """Extract text from parse tree."""
        for child in node.children:
            if child.symbol == "TEXT":
                return self._extract_value_from_node(child)
        return ""
    
    def _extract_url(self, node: ParseNode) -> str:
        """Extract URL from parse tree."""
        for child in node.children:
            if child.symbol == "URL":
                return self._extract_value_from_node(child)
        return ""
    
    def _extract_duration(self, node: ParseNode) -> float:
        """Extract duration from parse tree."""
        for child in node.children:
            if child.symbol == "DURATION":
                return self._extract_duration_from_node(child)
        return 1.0
    
    def _extract_value_from_node(self, node: ParseNode) -> str:
        """Extract string value from node."""
        if node.value:
            return node.value
        for child in node.children:
            if child.value:
                return child.value
        return ""
    
    def _extract_duration_from_node(self, node: ParseNode) -> float:
        """Extract duration value from node."""
        number = 1.0
        unit_multiplier = 1.0
        
        for child in node.children:
            if child.symbol == "NUMBER" and child.value:
                try:
                    number = float(child.value)
                except ValueError:
                    pass
            elif child.symbol == "TIME_UNIT" and child.value:
                unit = child.value.lower()
                if 'minute' in unit or 'min' in unit:
                    unit_multiplier = 60.0
                elif 'hour' in unit or 'hr' in unit:
                    unit_multiplier = 3600.0
        
        return number * unit_multiplier
    
    def _generate_token_id(self) -> int:
        """Generate unique token ID."""
        import hashlib
        import time
        seed = f"{time.time()}_{id(self)}"
        return abs(hash(seed)) % (2**31)


@dataclass
class ChartItem:
    """Item in the chart for Earley parsing."""
    production: Production
    dot_position: int
    start: int
    end: int
    parsed_tokens: List[ParsedToken] = field(default_factory=list)
    children: List['ChartItem'] = field(default_factory=list)
    
    def is_complete(self) -> bool:
        """Check if item is complete (dot at end)."""
        return self.dot_position >= len(self.production.rhs)
    
    def next_symbol(self) -> Optional[str]:
        """Get the symbol after the dot."""
        if self.dot_position < len(self.production.rhs):
            return self.production.rhs[self.dot_position]
        return None
    
    def next_symbol_is_terminal(self) -> bool:
        """Check if next symbol is terminal."""
        next_sym = self.next_symbol()
        return next_sym is not None and next_sym.isupper()
    
    def __eq__(self, other):
        if not isinstance(other, ChartItem):
            return False
        return (self.production == other.production and
                self.dot_position == other.dot_position and
                self.start == other.start and
                self.end == other.end)
    
    def __hash__(self):
        return hash((str(self.production.lhs), 
                    str(self.production.rhs),
                    self.dot_position, 
                    self.start, 
                    self.end))


# Grammar-based tokenizer integration
class GrammarTokenizer:
    """Tokenizer that uses formal grammar for parsing."""
    
    def __init__(self, abi: ActionABI = None):
        self.grammar = ActionGrammar(abi)
        self.parser = ChartParser(self.grammar)
    
    def tokenize_with_grammar(self, text: str) -> List[ActionToken]:
        """Tokenize text using grammar-based parsing."""
        # Simple tokenization (would be more sophisticated in practice)
        tokens = text.lower().split()
        
        # Parse with grammar
        parse_trees = self.parser.parse(tokens)
        
        # Extract action tokens from best parse
        if parse_trees:
            return self._extract_tokens_from_parse(parse_trees[0])
        
        return []
    
    def _extract_tokens_from_parse(self, parse_tree: ParseNode) -> List[ActionToken]:
        """Extract action tokens from parse tree."""
        tokens = []
        
        def traverse(node):
            if node.semantic_value and isinstance(node.semantic_value, ActionToken):
                tokens.append(node.semantic_value)
            for child in node.children:
                traverse(child)
        
        traverse(parse_tree)
        return tokens
