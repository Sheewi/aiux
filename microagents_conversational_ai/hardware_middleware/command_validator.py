"""
Command Validator - Security and validation for hardware commands
Validates, sanitizes, and controls hardware commands before execution.
"""

import re
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Command risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationResult(Enum):
    """Command validation results."""
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_APPROVAL = "requires_approval"
    MODIFIED = "modified"

@dataclass
class CommandValidationRequest:
    """Request for command validation."""
    command_id: str
    device_id: str
    command_type: str
    command: str
    args: Dict[str, Any]
    source: str
    timestamp: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class CommandValidationResponse:
    """Response from command validation."""
    request_id: str
    result: ValidationResult
    risk_level: RiskLevel
    modified_command: Optional[str] = None
    modified_args: Optional[Dict[str, Any]] = None
    reason: str = ""
    requires_approval: bool = False
    approval_message: str = ""
    estimated_impact: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'result': self.result.value,
            'risk_level': self.risk_level.value,
            'modified_command': self.modified_command,
            'modified_args': self.modified_args,
            'reason': self.reason,
            'requires_approval': self.requires_approval,
            'approval_message': self.approval_message,
            'estimated_impact': self.estimated_impact
        }

class CommandRule:
    """Represents a command validation rule."""
    
    def __init__(self, name: str, pattern: str, risk_level: RiskLevel, 
                 action: str = "allow", message: str = ""):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.risk_level = risk_level
        self.action = action  # allow, deny, require_approval, modify
        self.message = message
    
    def matches(self, command: str) -> bool:
        """Check if command matches this rule."""
        return bool(self.pattern.search(command))

class DevicePolicy:
    """Security policy for a specific device type."""
    
    def __init__(self, device_type: str):
        self.device_type = device_type
        self.rules: List[CommandRule] = []
        self.allowed_commands: Set[str] = set()
        self.denied_commands: Set[str] = set()
        self.requires_approval: Set[str] = set()
        self.max_command_length = 1000
        self.rate_limit_per_minute = 60
        
    def add_rule(self, rule: CommandRule):
        """Add a validation rule."""
        self.rules.append(rule)
    
    def add_allowed_command(self, command: str):
        """Add an explicitly allowed command."""
        self.allowed_commands.add(command.lower())
    
    def add_denied_command(self, command: str):
        """Add an explicitly denied command."""
        self.denied_commands.add(command.lower())
    
    def add_approval_required_command(self, command: str):
        """Add a command that requires approval."""
        self.requires_approval.add(command.lower())

class CommandValidator:
    """Central command validation and security service."""
    
    def __init__(self):
        self.policies: Dict[str, DevicePolicy] = {}
        self.global_rules: List[CommandRule] = []
        self.approval_callbacks: List[Callable[[CommandValidationRequest], bool]] = []
        self.audit_log: List[Dict[str, Any]] = []
        self.rate_limits: Dict[str, List[float]] = {}  # source -> timestamps
        
        # Initialize default policies and rules
        self._setup_default_policies()
        self._setup_default_rules()
    
    def _setup_default_policies(self):
        """Setup default security policies for common device types."""
        
        # USB/Serial device policy
        usb_policy = DevicePolicy("usb")
        usb_policy.add_rule(CommandRule(
            "dangerous_usb", r"(format|erase|reset|flash|burn)",
            RiskLevel.HIGH, "require_approval",
            "USB command may modify device firmware"
        ))
        self.policies["usb"] = usb_policy
        
        # Network device policy
        network_policy = DevicePolicy("network")
        network_policy.add_rule(CommandRule(
            "network_config", r"(ifconfig|ip\s+route|iptables|firewall)",
            RiskLevel.MEDIUM, "require_approval",
            "Network configuration changes require approval"
        ))
        self.policies["network"] = network_policy
        
        # Storage device policy
        storage_policy = DevicePolicy("storage")
        storage_policy.add_rule(CommandRule(
            "destructive_storage", r"(dd|fdisk|mkfs|format|rm\s+-rf)",
            RiskLevel.CRITICAL, "deny",
            "Destructive storage operations are not allowed"
        ))
        storage_policy.add_rule(CommandRule(
            "mount_unmount", r"(mount|umount)",
            RiskLevel.MEDIUM, "require_approval",
            "Mount operations require approval"
        ))
        self.policies["storage"] = storage_policy
        
        # Camera device policy
        camera_policy = DevicePolicy("camera")
        camera_policy.add_rule(CommandRule(
            "privacy_concern", r"(record|capture|stream)",
            RiskLevel.MEDIUM, "require_approval",
            "Camera operations may have privacy implications"
        ))
        self.policies["camera"] = camera_policy
        
        # Audio device policy
        audio_policy = DevicePolicy("audio")
        audio_policy.add_rule(CommandRule(
            "audio_capture", r"(record|capture|listen)",
            RiskLevel.MEDIUM, "require_approval",
            "Audio recording may have privacy implications"
        ))
        self.policies["audio"] = audio_policy
    
    def _setup_default_rules(self):
        """Setup global security rules."""
        
        # Dangerous system commands
        self.global_rules.extend([
            CommandRule(
                "system_shutdown", r"(shutdown|reboot|halt|poweroff)",
                RiskLevel.HIGH, "require_approval",
                "System shutdown/reboot requires approval"
            ),
            CommandRule(
                "user_management", r"(useradd|userdel|passwd|sudo|su)",
                RiskLevel.HIGH, "deny",
                "User management commands are not allowed"
            ),
            CommandRule(
                "file_system_damage", r"(rm\s+-rf\s+/|mkfs|fdisk)",
                RiskLevel.CRITICAL, "deny",
                "Potentially destructive filesystem operations are not allowed"
            ),
            CommandRule(
                "network_attacks", r"(nmap|nikto|sqlmap|metasploit)",
                RiskLevel.CRITICAL, "deny",
                "Security testing tools are not allowed"
            ),
            CommandRule(
                "process_manipulation", r"(kill\s+-9|killall|pkill.*-f)",
                RiskLevel.MEDIUM, "require_approval",
                "Process termination requires approval"
            ),
            CommandRule(
                "service_control", r"(systemctl|service|chkconfig)",
                RiskLevel.MEDIUM, "require_approval",
                "Service control requires approval"
            ),
            CommandRule(
                "package_management", r"(apt|yum|dnf|pip|npm)\s+(install|remove|uninstall)",
                RiskLevel.MEDIUM, "require_approval",
                "Package installation/removal requires approval"
            )
        ])
    
    def add_policy(self, device_type: str, policy: DevicePolicy):
        """Add a device policy."""
        self.policies[device_type] = policy
        logger.info(f"Added policy for device type: {device_type}")
    
    def add_global_rule(self, rule: CommandRule):
        """Add a global validation rule."""
        self.global_rules.append(rule)
        logger.info(f"Added global rule: {rule.name}")
    
    def add_approval_callback(self, callback: Callable[[CommandValidationRequest], bool]):
        """Add a callback for approval requests."""
        self.approval_callbacks.append(callback)
    
    def validate_command(self, request: CommandValidationRequest) -> CommandValidationResponse:
        """Validate a command request."""
        start_time = time.time()
        
        # Check rate limiting
        if not self._check_rate_limit(request.source):
            return CommandValidationResponse(
                request_id=request.command_id,
                result=ValidationResult.REJECTED,
                risk_level=RiskLevel.MEDIUM,
                reason="Rate limit exceeded",
                estimated_impact="Request denied due to too many commands"
            )
        
        # Basic input validation
        basic_check = self._basic_validation(request)
        if basic_check:
            return basic_check
        
        # Apply global rules
        for rule in self.global_rules:
            if rule.matches(request.command):
                response = self._apply_rule(request, rule)
                if response:
                    self._log_validation(request, response, time.time() - start_time)
                    return response
        
        # Apply device-specific policy
        device_type = self._determine_device_type(request.device_id)
        if device_type in self.policies:
            policy = self.policies[device_type]
            
            # Check explicit lists first
            command_lower = request.command.lower()
            
            if command_lower in policy.denied_commands:
                response = CommandValidationResponse(
                    request_id=request.command_id,
                    result=ValidationResult.REJECTED,
                    risk_level=RiskLevel.HIGH,
                    reason=f"Command explicitly denied for {device_type} devices",
                    estimated_impact="Command execution blocked by policy"
                )
                self._log_validation(request, response, time.time() - start_time)
                return response
            
            if command_lower in policy.requires_approval:
                response = CommandValidationResponse(
                    request_id=request.command_id,
                    result=ValidationResult.REQUIRES_APPROVAL,
                    risk_level=RiskLevel.MEDIUM,
                    requires_approval=True,
                    approval_message=f"Command requires approval for {device_type} device",
                    estimated_impact="Command will execute after approval"
                )
                self._log_validation(request, response, time.time() - start_time)
                return response
            
            if command_lower in policy.allowed_commands:
                response = CommandValidationResponse(
                    request_id=request.command_id,
                    result=ValidationResult.APPROVED,
                    risk_level=RiskLevel.LOW,
                    reason=f"Command explicitly allowed for {device_type} devices",
                    estimated_impact="Command will execute immediately"
                )
                self._log_validation(request, response, time.time() - start_time)
                return response
            
            # Apply policy rules
            for rule in policy.rules:
                if rule.matches(request.command):
                    response = self._apply_rule(request, rule)
                    if response:
                        self._log_validation(request, response, time.time() - start_time)
                        return response
        
        # Default: allow with low risk if no rules matched
        response = CommandValidationResponse(
            request_id=request.command_id,
            result=ValidationResult.APPROVED,
            risk_level=RiskLevel.LOW,
            reason="No matching security rules, allowing command",
            estimated_impact="Command will execute with standard monitoring"
        )
        
        self._log_validation(request, response, time.time() - start_time)
        return response
    
    def _basic_validation(self, request: CommandValidationRequest) -> Optional[CommandValidationResponse]:
        """Perform basic validation checks."""
        
        # Check command length
        if len(request.command) > 1000:
            return CommandValidationResponse(
                request_id=request.command_id,
                result=ValidationResult.REJECTED,
                risk_level=RiskLevel.MEDIUM,
                reason="Command too long",
                estimated_impact="Overly long commands are rejected for security"
            )
        
        # Check for null bytes or control characters
        if '\x00' in request.command or any(ord(c) < 32 and c not in '\t\n\r' for c in request.command):
            return CommandValidationResponse(
                request_id=request.command_id,
                result=ValidationResult.REJECTED,
                risk_level=RiskLevel.HIGH,
                reason="Command contains invalid characters",
                estimated_impact="Commands with control characters are rejected"
            )
        
        # Check for command injection patterns
        injection_patterns = [
            r';.*rm\s+-rf',
            r'\|\s*rm\s+-rf',
            r'&&.*rm\s+-rf',
            r'\$\(.*\)',
            r'`.*`',
            r'>\s*/dev/',
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, request.command, re.IGNORECASE):
                return CommandValidationResponse(
                    request_id=request.command_id,
                    result=ValidationResult.REJECTED,
                    risk_level=RiskLevel.CRITICAL,
                    reason="Potential command injection detected",
                    estimated_impact="Command appears to contain injection attempt"
                )
        
        return None
    
    def _apply_rule(self, request: CommandValidationRequest, rule: CommandRule) -> Optional[CommandValidationResponse]:
        """Apply a validation rule to a command."""
        
        if rule.action == "deny":
            return CommandValidationResponse(
                request_id=request.command_id,
                result=ValidationResult.REJECTED,
                risk_level=rule.risk_level,
                reason=f"Rule '{rule.name}': {rule.message}",
                estimated_impact="Command blocked by security rule"
            )
        
        elif rule.action == "require_approval":
            return CommandValidationResponse(
                request_id=request.command_id,
                result=ValidationResult.REQUIRES_APPROVAL,
                risk_level=rule.risk_level,
                requires_approval=True,
                approval_message=f"Rule '{rule.name}': {rule.message}",
                estimated_impact="Command requires manual approval to proceed"
            )
        
        elif rule.action == "modify":
            # Apply command modifications (implement specific modifications as needed)
            modified_command = self._apply_command_modifications(request.command, rule)
            return CommandValidationResponse(
                request_id=request.command_id,
                result=ValidationResult.MODIFIED,
                risk_level=rule.risk_level,
                modified_command=modified_command,
                reason=f"Rule '{rule.name}': Command modified for safety",
                estimated_impact="Command was automatically modified for security"
            )
        
        # rule.action == "allow" or unknown action
        return None
    
    def _apply_command_modifications(self, command: str, rule: CommandRule) -> str:
        """Apply safety modifications to a command."""
        # Example modifications (extend as needed)
        
        # Add safety flags to potentially dangerous commands
        if 'rm ' in command and '-rf' not in command:
            command = command.replace('rm ', 'rm -i ')  # Add interactive flag
        
        if 'cp ' in command and '-i' not in command:
            command = command.replace('cp ', 'cp -i ')  # Add interactive flag
        
        if 'mv ' in command and '-i' not in command:
            command = command.replace('mv ', 'mv -i ')  # Add interactive flag
        
        return command
    
    def _check_rate_limit(self, source: str, max_per_minute: int = 60) -> bool:
        """Check if source is within rate limits."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        if source not in self.rate_limits:
            self.rate_limits[source] = []
        
        # Remove old timestamps
        self.rate_limits[source] = [
            ts for ts in self.rate_limits[source] if ts > minute_ago
        ]
        
        # Check limit
        if len(self.rate_limits[source]) >= max_per_minute:
            return False
        
        # Add current timestamp
        self.rate_limits[source].append(current_time)
        return True
    
    def _determine_device_type(self, device_id: str) -> str:
        """Determine device type from device ID."""
        if device_id.startswith('usb_'):
            return 'usb'
        elif device_id.startswith('network_'):
            return 'network'
        elif device_id.startswith('storage_'):
            return 'storage'
        elif device_id.startswith('camera_'):
            return 'camera'
        elif device_id.startswith('audio_'):
            return 'audio'
        else:
            return 'unknown'
    
    def _log_validation(self, request: CommandValidationRequest, 
                       response: CommandValidationResponse, duration: float):
        """Log validation results for audit."""
        log_entry = {
            'timestamp': time.time(),
            'request_id': request.command_id,
            'device_id': request.device_id,
            'command': request.command,
            'source': request.source,
            'result': response.result.value,
            'risk_level': response.risk_level.value,
            'reason': response.reason,
            'duration_ms': duration * 1000,
            'user_id': request.user_id,
            'session_id': request.session_id
        }
        
        self.audit_log.append(log_entry)
        
        # Keep only last 10000 entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
        
        # Log to standard logger as well
        logger.info(f"Command validation: {response.result.value} - {request.device_id} - {request.command[:50]}")
    
    def request_approval(self, request: CommandValidationRequest) -> bool:
        """Request approval for a command through callbacks."""
        for callback in self.approval_callbacks:
            try:
                if callback(request):
                    logger.info(f"Command approved: {request.command_id}")
                    return True
            except Exception as e:
                logger.error(f"Approval callback error: {e}")
        
        logger.warning(f"Command not approved: {request.command_id}")
        return False
    
    def get_audit_log(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get audit log entries from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [entry for entry in self.audit_log if entry['timestamp'] > cutoff_time]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        recent_log = self.get_audit_log(24)
        
        if not recent_log:
            return {'total': 0}
        
        total = len(recent_log)
        by_result = {}
        by_risk = {}
        by_device = {}
        
        for entry in recent_log:
            result = entry['result']
            risk = entry['risk_level']
            device = entry['device_id']
            
            by_result[result] = by_result.get(result, 0) + 1
            by_risk[risk] = by_risk.get(risk, 0) + 1
            by_device[device] = by_device.get(device, 0) + 1
        
        return {
            'total': total,
            'by_result': by_result,
            'by_risk_level': by_risk,
            'by_device': by_device,
            'approval_rate': by_result.get('approved', 0) / total if total > 0 else 0
        }
    
    def export_audit_log(self, filename: str, hours: int = 24):
        """Export audit log to file."""
        audit_data = {
            'export_timestamp': time.time(),
            'hours': hours,
            'entries': self.get_audit_log(hours)
        }
        
        with open(filename, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        logger.info(f"Exported {len(audit_data['entries'])} audit entries to {filename}")

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def approval_callback(request: CommandValidationRequest) -> bool:
        # Simple approval for demo - in practice, this would involve user interaction
        print(f"Approval requested for: {request.command}")
        print(f"Device: {request.device_id}, Source: {request.source}")
        
        # Auto-approve non-critical commands for demo
        return True
    
    # Create validator
    validator = CommandValidator()
    validator.add_approval_callback(approval_callback)
    
    # Test commands
    test_commands = [
        # Safe commands
        CommandValidationRequest("1", "camera_0", "capture", "capture_image", {}, "ai_agent", time.time()),
        CommandValidationRequest("2", "usb_device1", "query", "get_status", {}, "ai_agent", time.time()),
        
        # Risky commands  
        CommandValidationRequest("3", "system", "shutdown", "shutdown -h now", {}, "ai_agent", time.time()),
        CommandValidationRequest("4", "storage_sda1", "format", "mkfs.ext4 /dev/sda1", {}, "ai_agent", time.time()),
        
        # Malicious commands
        CommandValidationRequest("5", "system", "exec", "rm -rf /", {}, "unknown", time.time()),
        CommandValidationRequest("6", "system", "exec", "ls; rm -rf /tmp", {}, "unknown", time.time()),
    ]
    
    print("Testing command validation:")
    print("=" * 50)
    
    for request in test_commands:
        response = validator.validate_command(request)
        print(f"\nCommand: {request.command}")
        print(f"Result: {response.result.value}")
        print(f"Risk: {response.risk_level.value}")
        print(f"Reason: {response.reason}")
        
        if response.requires_approval:
            print("  -> Requesting approval...")
            approved = validator.request_approval(request)
            print(f"  -> Approval: {'GRANTED' if approved else 'DENIED'}")
    
    # Show statistics
    print("\n" + "=" * 50)
    print("Validation Statistics:")
    stats = validator.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Export audit log
    validator.export_audit_log("command_audit.json")
