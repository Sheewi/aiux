import asyncio
import hashlib
import hmac
import secrets
import jwt
import base64
import logging
import re
import ssl
import socket
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

from .base_tool import BaseTool, ToolStatus, ToolMetadata, ToolType, ToolCapability, create_tool_metadata


class SecurityTool(BaseTool):
    """Comprehensive security tool for encryption, authentication, and security analysis."""
    
    def __init__(self, secret_key: Optional[str] = None, config: Dict[str, Any] = None):
        """
        Initialize the security tool.
        
        Args:
            secret_key: Secret key for encryption operations
            config: Additional configuration
        """
        # Initialize metadata
        metadata = create_tool_metadata(
            tool_id="security",
            name="Security Tool",
            description="Comprehensive security tool for encryption, hashing, password generation, authentication, and vulnerability analysis",
            tool_type=ToolType.SECURITY,
            version="1.0.0",
            author="System",
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.STATEFUL,
                ToolCapability.REQUIRES_AUTH
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["encrypt", "decrypt", "hash", "generate_password", "password_strength", "scan_vulnerability"]},
                    "data": {"type": "string", "description": "Data to process"},
                    "algorithm": {"type": "string", "description": "Algorithm to use"},
                    "key": {"type": "string", "description": "Encryption/decryption key"},
                    "length": {"type": "integer", "description": "Password length"},
                    "include_symbols": {"type": "boolean", "description": "Include symbols in password"}
                },
                "required": ["operation"]
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
            supported_formats=["text", "binary", "base64"],
            tags=["security", "encryption", "cryptography", "authentication", "vulnerability"]
        )
        
        super().__init__(metadata, config)
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute security operation.
        
        Args:
            operation: Type of security operation
            **kwargs: Operation-specific parameters
        """
        self.status = ToolStatus.RUNNING
        
        try:
            operation_map = {
                'hash': self._hash_data,
                'encrypt': self._encrypt_data,
                'decrypt': self._decrypt_data,
                'generate_token': self._generate_token,
                'verify_token': self._verify_token,
                'password_strength': self._check_password_strength,
                'generate_password': self._generate_password,
                'scan_vulnerabilities': self._scan_vulnerabilities,
                'validate_ssl': self._validate_ssl_certificate,
                'sanitize_input': self._sanitize_input,
                'audit_permissions': self._audit_permissions,
                'detect_threats': self._detect_threats
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
            self.logger.error(f"Security operation failed: {e}")
            return {
                'success': False,
                'operation': operation,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _hash_data(self, data: str, 
                        algorithm: str = 'sha256',
                        salt: Optional[str] = None) -> Dict[str, Any]:
        """Generate hash of data."""
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        # Add salt if provided
        if salt:
            data = salt + data
        
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode('utf-8'))
        hash_value = hash_obj.hexdigest()
        
        return {
            'algorithm': algorithm,
            'hash': hash_value,
            'salt_used': bool(salt),
            'input_length': len(data)
        }
    
    async def _encrypt_data(self, data: str, 
                           method: str = 'fernet',
                           password: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt data using various methods."""
        if method == 'fernet':
            if password:
                # Derive key from password
                password_bytes = password.encode()
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            else:
                key = Fernet.generate_key()
                salt = None
            
            fernet = Fernet(key)
            encrypted_data = fernet.encrypt(data.encode('utf-8'))
            
            return {
                'method': method,
                'encrypted_data': base64.b64encode(encrypted_data).decode('ascii'),
                'key': key.decode('ascii') if not password else None,
                'salt': base64.b64encode(salt).decode('ascii') if salt else None,
                'password_protected': bool(password)
            }
        
        elif method == 'base64':
            # Simple base64 encoding (not secure encryption)
            encoded_data = base64.b64encode(data.encode('utf-8')).decode('ascii')
            return {
                'method': method,
                'encrypted_data': encoded_data,
                'key': None,
                'salt': None,
                'password_protected': False,
                'warning': 'Base64 is encoding, not encryption. Not secure for sensitive data.'
            }
        
        else:
            raise ValueError(f"Unsupported encryption method: {method}")
    
    async def _decrypt_data(self, encrypted_data: str,
                           method: str = 'fernet',
                           key: Optional[str] = None,
                           password: Optional[str] = None,
                           salt: Optional[str] = None) -> Dict[str, Any]:
        """Decrypt data."""
        if method == 'fernet':
            if password and salt:
                # Derive key from password and salt
                password_bytes = password.encode()
                salt_bytes = base64.b64decode(salt.encode('ascii'))
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt_bytes,
                    iterations=100000,
                )
                derived_key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            elif key:
                derived_key = key.encode('ascii')
            else:
                raise ValueError("Either key or (password + salt) required for decryption")
            
            fernet = Fernet(derived_key)
            encrypted_bytes = base64.b64decode(encrypted_data.encode('ascii'))
            decrypted_data = fernet.decrypt(encrypted_bytes)
            
            return {
                'method': method,
                'decrypted_data': decrypted_data.decode('utf-8'),
                'success': True
            }
        
        elif method == 'base64':
            # Simple base64 decoding
            decoded_data = base64.b64decode(encrypted_data.encode('ascii')).decode('utf-8')
            return {
                'method': method,
                'decrypted_data': decoded_data,
                'success': True
            }
        
        else:
            raise ValueError(f"Unsupported decryption method: {method}")
    
    async def _generate_token(self, payload: Dict[str, Any],
                             token_type: str = 'jwt',
                             expires_in: int = 3600,
                             secret: Optional[str] = None) -> Dict[str, Any]:
        """Generate authentication token."""
        if token_type == 'jwt':
            # JWT token generation
            secret_key = secret or self.secret_key
            
            # Add standard claims
            now = datetime.utcnow()
            payload.update({
                'iat': now,
                'exp': now + timedelta(seconds=expires_in),
                'iss': 'security_tool'
            })
            
            token = jwt.encode(payload, secret_key, algorithm='HS256')
            
            return {
                'token_type': token_type,
                'token': token,
                'expires_in': expires_in,
                'payload': payload
            }
        
        elif token_type == 'random':
            # Random token generation
            token = secrets.token_urlsafe(32)
            
            return {
                'token_type': token_type,
                'token': token,
                'expires_in': expires_in,
                'payload': payload
            }
        
        else:
            raise ValueError(f"Unsupported token type: {token_type}")
    
    async def _verify_token(self, token: str,
                           token_type: str = 'jwt',
                           secret: Optional[str] = None) -> Dict[str, Any]:
        """Verify authentication token."""
        if token_type == 'jwt':
            secret_key = secret or self.secret_key
            
            try:
                payload = jwt.decode(token, secret_key, algorithms=['HS256'])
                
                return {
                    'valid': True,
                    'payload': payload,
                    'expired': False,
                    'token_type': token_type
                }
            
            except jwt.ExpiredSignatureError:
                return {
                    'valid': False,
                    'payload': None,
                    'expired': True,
                    'error': 'Token has expired',
                    'token_type': token_type
                }
            
            except jwt.InvalidTokenError as e:
                return {
                    'valid': False,
                    'payload': None,
                    'expired': False,
                    'error': str(e),
                    'token_type': token_type
                }
        
        else:
            # For random tokens, just check format
            return {
                'valid': bool(token and len(token) >= 16),
                'payload': None,
                'expired': False,
                'token_type': token_type
            }
    
    async def _check_password_strength(self, password: str) -> Dict[str, Any]:
        """Analyze password strength."""
        strength_score = 0
        feedback = []
        
        # Length check
        if len(password) >= 8:
            strength_score += 1
        else:
            feedback.append("Password should be at least 8 characters long")
        
        if len(password) >= 12:
            strength_score += 1
        
        # Character variety checks
        if re.search(r'[a-z]', password):
            strength_score += 1
        else:
            feedback.append("Include lowercase letters")
        
        if re.search(r'[A-Z]', password):
            strength_score += 1
        else:
            feedback.append("Include uppercase letters")
        
        if re.search(r'\d', password):
            strength_score += 1
        else:
            feedback.append("Include numbers")
        
        if re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', password):
            strength_score += 1
        else:
            feedback.append("Include special characters")
        
        # Common password patterns
        common_patterns = [
            r'123',
            r'abc',
            r'password',
            r'qwerty',
            r'admin'
        ]
        
        for pattern in common_patterns:
            if re.search(pattern, password.lower()):
                strength_score -= 1
                feedback.append(f"Avoid common patterns like '{pattern}'")
        
        # Determine strength level
        if strength_score >= 5:
            strength_level = "Strong"
        elif strength_score >= 3:
            strength_level = "Medium"
        else:
            strength_level = "Weak"
        
        return {
            'password_length': len(password),
            'strength_score': max(0, strength_score),
            'max_score': 6,
            'strength_level': strength_level,
            'feedback': feedback,
            'estimated_crack_time': self._estimate_crack_time(password)
        }
    
    def _estimate_crack_time(self, password: str) -> str:
        """Estimate password crack time."""
        # Simple estimation based on character set and length
        charset_size = 0
        
        if re.search(r'[a-z]', password):
            charset_size += 26
        if re.search(r'[A-Z]', password):
            charset_size += 26
        if re.search(r'\d', password):
            charset_size += 10
        if re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', password):
            charset_size += 32
        
        if charset_size == 0:
            return "Unknown"
        
        # Calculate possible combinations
        combinations = charset_size ** len(password)
        
        # Assume 1 billion guesses per second
        guesses_per_second = 1e9
        seconds_to_crack = combinations / (2 * guesses_per_second)  # Average case
        
        if seconds_to_crack < 60:
            return "Less than 1 minute"
        elif seconds_to_crack < 3600:
            return f"{int(seconds_to_crack / 60)} minutes"
        elif seconds_to_crack < 86400:
            return f"{int(seconds_to_crack / 3600)} hours"
        elif seconds_to_crack < 31536000:
            return f"{int(seconds_to_crack / 86400)} days"
        else:
            return f"{int(seconds_to_crack / 31536000)} years"
    
    async def _generate_password(self, length: int = 12,
                                include_uppercase: bool = True,
                                include_lowercase: bool = True,
                                include_numbers: bool = True,
                                include_symbols: bool = True,
                                exclude_ambiguous: bool = True) -> Dict[str, Any]:
        """Generate secure password."""
        characters = ""
        
        if include_lowercase:
            chars = "abcdefghijklmnopqrstuvwxyz"
            if exclude_ambiguous:
                chars = chars.replace('l', '').replace('o', '')
            characters += chars
        
        if include_uppercase:
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if exclude_ambiguous:
                chars = chars.replace('I', '').replace('O', '')
            characters += chars
        
        if include_numbers:
            chars = "0123456789"
            if exclude_ambiguous:
                chars = chars.replace('0', '').replace('1', '')
            characters += chars
        
        if include_symbols:
            chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            characters += chars
        
        if not characters:
            raise ValueError("At least one character type must be included")
        
        # Generate password
        password = ''.join(secrets.choice(characters) for _ in range(length))
        
        # Check strength
        strength_check = await self._check_password_strength(password)
        
        return {
            'password': password,
            'length': length,
            'character_types': {
                'uppercase': include_uppercase,
                'lowercase': include_lowercase,
                'numbers': include_numbers,
                'symbols': include_symbols
            },
            'strength_analysis': strength_check
        }
    
    async def _scan_vulnerabilities(self, target_type: str,
                                   target: str,
                                   scan_depth: str = 'basic') -> Dict[str, Any]:
        """Scan for common vulnerabilities."""
        vulnerabilities = []
        
        if target_type == 'url':
            # Basic URL vulnerability checks
            vulnerabilities.extend(await self._check_url_vulnerabilities(target, scan_depth))
        
        elif target_type == 'code':
            # Basic code vulnerability checks
            vulnerabilities.extend(await self._check_code_vulnerabilities(target, scan_depth))
        
        elif target_type == 'config':
            # Configuration vulnerability checks
            vulnerabilities.extend(await self._check_config_vulnerabilities(target, scan_depth))
        
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        # Categorize vulnerabilities by severity
        severity_counts = {'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        for vuln in vulnerabilities:
            severity_counts[vuln.get('severity', 'info')] += 1
        
        return {
            'target_type': target_type,
            'target': target,
            'scan_depth': scan_depth,
            'vulnerabilities': vulnerabilities,
            'summary': {
                'total_issues': len(vulnerabilities),
                'severity_breakdown': severity_counts,
                'scan_time': datetime.now().isoformat()
            }
        }
    
    async def _check_url_vulnerabilities(self, url: str, scan_depth: str) -> List[Dict[str, Any]]:
        """Check URL for vulnerabilities."""
        vulnerabilities = []
        
        # Check for HTTP instead of HTTPS
        if url.startswith('http://'):
            vulnerabilities.append({
                'type': 'insecure_transport',
                'severity': 'medium',
                'description': 'URL uses HTTP instead of HTTPS',
                'recommendation': 'Use HTTPS for secure communication'
            })
        
        # Check for common vulnerable parameters
        vulnerable_params = ['id', 'user', 'admin', 'debug', 'test']
        for param in vulnerable_params:
            if f'{param}=' in url:
                vulnerabilities.append({
                    'type': 'potentially_vulnerable_parameter',
                    'severity': 'low',
                    'description': f'URL contains potentially vulnerable parameter: {param}',
                    'recommendation': 'Validate and sanitize all URL parameters'
                })
        
        # Check for SQL injection patterns
        sql_patterns = ['SELECT', 'UNION', 'DROP', 'INSERT', 'UPDATE', 'DELETE']
        for pattern in sql_patterns:
            if pattern.upper() in url.upper():
                vulnerabilities.append({
                    'type': 'potential_sql_injection',
                    'severity': 'high',
                    'description': f'URL contains SQL keyword: {pattern}',
                    'recommendation': 'Use parameterized queries and input validation'
                })
        
        return vulnerabilities
    
    async def _check_code_vulnerabilities(self, code: str, scan_depth: str) -> List[Dict[str, Any]]:
        """Check code for vulnerabilities."""
        vulnerabilities = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded_password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'hardcoded_api_key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'hardcoded_secret'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'hardcoded_token')
        ]
        
        for pattern, vuln_type in secret_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                vulnerabilities.append({
                    'type': vuln_type,
                    'severity': 'high',
                    'description': f'Hardcoded secret found: {match.group()}',
                    'recommendation': 'Use environment variables or secure vaults for secrets'
                })
        
        # Check for SQL injection vulnerabilities
        sql_injection_patterns = [
            r'execute\s*\(\s*["\'][^"\']*\+',
            r'query\s*\(\s*["\'][^"\']*\+',
            r'SELECT\s+.*\+.*FROM'
        ]
        
        for pattern in sql_injection_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'sql_injection_risk',
                    'severity': 'high',
                    'description': 'Potential SQL injection vulnerability found',
                    'recommendation': 'Use parameterized queries or ORM'
                })
        
        # Check for XSS vulnerabilities
        xss_patterns = [
            r'innerHTML\s*=',
            r'document\.write\s*\(',
            r'eval\s*\('
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'xss_risk',
                    'severity': 'medium',
                    'description': 'Potential XSS vulnerability found',
                    'recommendation': 'Sanitize and validate all user inputs'
                })
        
        return vulnerabilities
    
    async def _check_config_vulnerabilities(self, config: str, scan_depth: str) -> List[Dict[str, Any]]:
        """Check configuration for vulnerabilities."""
        vulnerabilities = []
        
        # Parse config as JSON or key-value pairs
        try:
            if config.strip().startswith('{'):
                config_data = json.loads(config)
            else:
                # Simple key=value format
                config_data = {}
                for line in config.split('\n'):
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        config_data[key.strip()] = value.strip()
        except:
            config_data = {}
        
        # Check for insecure configurations
        insecure_configs = {
            'debug': {'values': ['true', '1', 'on'], 'severity': 'medium'},
            'ssl_verify': {'values': ['false', '0', 'off'], 'severity': 'high'},
            'allow_all_origins': {'values': ['*', 'true'], 'severity': 'high'},
            'admin_password': {'values': ['admin', 'password', '123456'], 'severity': 'high'}
        }
        
        for key, value in config_data.items():
            key_lower = key.lower()
            value_lower = str(value).lower()
            
            for config_key, config_info in insecure_configs.items():
                if config_key in key_lower and value_lower in config_info['values']:
                    vulnerabilities.append({
                        'type': 'insecure_configuration',
                        'severity': config_info['severity'],
                        'description': f'Insecure configuration: {key}={value}',
                        'recommendation': f'Review and secure the {key} configuration'
                    })
        
        return vulnerabilities
    
    async def _validate_ssl_certificate(self, hostname: str, 
                                       port: int = 443) -> Dict[str, Any]:
        """Validate SSL certificate for a hostname."""
        try:
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect and get certificate
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
            
            # Extract certificate information
            subject = dict(x[0] for x in cert['subject'])
            issuer = dict(x[0] for x in cert['issuer'])
            
            # Check expiration
            not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
            days_until_expiry = (not_after - datetime.now()).days
            
            # Determine certificate status
            is_valid = True
            issues = []
            
            if days_until_expiry < 0:
                is_valid = False
                issues.append("Certificate has expired")
            elif days_until_expiry < 30:
                issues.append(f"Certificate expires in {days_until_expiry} days")
            
            # Check hostname match
            cert_hostnames = [subject.get('commonName', '')]
            if 'subjectAltName' in cert:
                cert_hostnames.extend([name[1] for name in cert['subjectAltName'] if name[0] == 'DNS'])
            
            hostname_match = any(
                hostname == cert_hostname or 
                (cert_hostname.startswith('*.') and hostname.endswith(cert_hostname[2:]))
                for cert_hostname in cert_hostnames
            )
            
            if not hostname_match:
                is_valid = False
                issues.append("Hostname does not match certificate")
            
            return {
                'valid': is_valid,
                'hostname': hostname,
                'port': port,
                'certificate_info': {
                    'subject': subject,
                    'issuer': issuer,
                    'not_before': cert['notBefore'],
                    'not_after': cert['notAfter'],
                    'days_until_expiry': days_until_expiry,
                    'serial_number': cert.get('serialNumber', ''),
                    'version': cert.get('version', '')
                },
                'issues': issues,
                'certificate_chain_length': len(cert.get('caIssuers', []))
            }
        
        except Exception as e:
            return {
                'valid': False,
                'hostname': hostname,
                'port': port,
                'error': str(e),
                'certificate_info': None
            }
    
    async def _sanitize_input(self, input_data: str,
                             sanitization_type: str = 'html') -> Dict[str, Any]:
        """Sanitize input data."""
        original_data = input_data
        
        if sanitization_type == 'html':
            # Basic HTML sanitization
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', input_data, flags=re.IGNORECASE | re.DOTALL)
            sanitized = re.sub(r'<.*?>', '', sanitized)
            sanitized = sanitized.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        
        elif sanitization_type == 'sql':
            # Basic SQL injection prevention
            sanitized = input_data.replace("'", "''")
            sanitized = re.sub(r'[;]', '', sanitized)
            sanitized = re.sub(r'(UNION|SELECT|INSERT|UPDATE|DELETE|DROP)', '', sanitized, flags=re.IGNORECASE)
        
        elif sanitization_type == 'xss':
            # XSS prevention
            sanitized = input_data.replace('<', '&lt;').replace('>', '&gt;')
            sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
            sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        else:
            # Basic sanitization
            sanitized = re.sub(r'[^\w\s\-\.\@]', '', input_data)
        
        return {
            'original_data': original_data,
            'sanitized_data': sanitized,
            'sanitization_type': sanitization_type,
            'characters_removed': len(original_data) - len(sanitized),
            'changes_made': original_data != sanitized
        }
    
    async def _audit_permissions(self, file_path: str) -> Dict[str, Any]:
        """Audit file/directory permissions."""
        try:
            import stat
            
            file_stat = os.stat(file_path)
            file_mode = file_stat.st_mode
            
            # Extract permission bits
            permissions = {
                'owner': {
                    'read': bool(file_mode & stat.S_IRUSR),
                    'write': bool(file_mode & stat.S_IWUSR),
                    'execute': bool(file_mode & stat.S_IXUSR)
                },
                'group': {
                    'read': bool(file_mode & stat.S_IRGRP),
                    'write': bool(file_mode & stat.S_IWGRP),
                    'execute': bool(file_mode & stat.S_IXGRP)
                },
                'others': {
                    'read': bool(file_mode & stat.S_IROTH),
                    'write': bool(file_mode & stat.S_IWOTH),
                    'execute': bool(file_mode & stat.S_IXOTH)
                }
            }
            
            # Security recommendations
            security_issues = []
            
            # Check for world-writable files
            if permissions['others']['write']:
                security_issues.append({
                    'severity': 'high',
                    'issue': 'File is world-writable',
                    'recommendation': 'Remove write permissions for others'
                })
            
            # Check for world-readable sensitive files
            if file_path.endswith(('.key', '.pem', '.p12', '.pfx')):
                if permissions['others']['read']:
                    security_issues.append({
                        'severity': 'high',
                        'issue': 'Sensitive file is world-readable',
                        'recommendation': 'Restrict read permissions to owner only'
                    })
            
            # Check for executable files that shouldn't be
            if file_path.endswith(('.txt', '.log', '.conf', '.ini')):
                if any(permissions[user]['execute'] for user in permissions):
                    security_issues.append({
                        'severity': 'medium',
                        'issue': 'Non-executable file has execute permissions',
                        'recommendation': 'Remove execute permissions'
                    })
            
            return {
                'file_path': file_path,
                'permissions': permissions,
                'octal_mode': oct(file_stat.st_mode)[-3:],
                'security_issues': security_issues,
                'owner_uid': file_stat.st_uid,
                'group_gid': file_stat.st_gid,
                'is_directory': stat.S_ISDIR(file_mode),
                'is_regular_file': stat.S_ISREG(file_mode)
            }
        
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'permissions': None
            }
    
    async def _detect_threats(self, data: str, 
                             threat_types: List[str] = None) -> Dict[str, Any]:
        """Detect potential security threats in data."""
        if threat_types is None:
            threat_types = ['malware_signatures', 'suspicious_patterns', 'data_exfiltration']
        
        detected_threats = []
        
        if 'malware_signatures' in threat_types:
            # Basic malware signature detection
            malware_patterns = [
                (r'eval\s*\(.*base64_decode', 'PHP obfuscated code'),
                (r'<script[^>]*>.*document\.cookie', 'Cookie stealing script'),
                (r'wget.*\|\s*sh', 'Suspicious download and execute'),
                (r'rm\s+-rf\s+/\*', 'Destructive command'),
                (r'nc\s+-l.*-e', 'Netcat backdoor')
            ]
            
            for pattern, description in malware_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    detected_threats.append({
                        'type': 'malware_signature',
                        'severity': 'high',
                        'description': description,
                        'pattern_matched': pattern
                    })
        
        if 'suspicious_patterns' in threat_types:
            # Suspicious pattern detection
            suspicious_patterns = [
                (r'password.*=.*["\'][^"\']{1,5}["\']', 'Weak password detected'),
                (r'\b\d{16}\b', 'Potential credit card number'),
                (r'\b\d{3}-\d{2}-\d{4}\b', 'Potential SSN'),
                (r'BEGIN\s+PRIVATE\s+KEY', 'Private key detected'),
                (r'[a-zA-Z0-9]{32,}', 'Potential API key or hash')
            ]
            
            for pattern, description in suspicious_patterns:
                matches = re.findall(pattern, data)
                if matches:
                    detected_threats.append({
                        'type': 'suspicious_pattern',
                        'severity': 'medium',
                        'description': description,
                        'matches_count': len(matches),
                        'pattern_matched': pattern
                    })
        
        if 'data_exfiltration' in threat_types:
            # Data exfiltration indicators
            exfil_patterns = [
                (r'curl.*-d.*http', 'Data being posted to external URL'),
                (r'ftp.*put.*', 'File upload to FTP'),
                (r'scp.*@.*:', 'Secure copy to remote host'),
                (r'email.*attach', 'Email with attachment')
            ]
            
            for pattern, description in exfil_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    detected_threats.append({
                        'type': 'data_exfiltration',
                        'severity': 'high',
                        'description': description,
                        'pattern_matched': pattern
                    })
        
        # Calculate threat level
        if any(threat['severity'] == 'high' for threat in detected_threats):
            threat_level = 'HIGH'
        elif any(threat['severity'] == 'medium' for threat in detected_threats):
            threat_level = 'MEDIUM'
        elif detected_threats:
            threat_level = 'LOW'
        else:
            threat_level = 'CLEAN'
        
        return {
            'threat_level': threat_level,
            'threats_detected': len(detected_threats),
            'threat_details': detected_threats,
            'scan_types': threat_types,
            'data_size': len(data),
            'scan_timestamp': datetime.now().isoformat()
        }
