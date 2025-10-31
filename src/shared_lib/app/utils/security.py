"""
Aura Platform - Security Utilities
Security utilities for authentication, authorization, and data protection.
"""

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import jwt
from cryptography.fernet import Fernet
import base64
from os import getenv
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """Security manager for authentication and authorization."""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
    ):
        """
        Initialize the security manager.
        
        Args:
            secret_key: The secret key for JWT signing
            algorithm: The JWT algorithm
            access_token_expire_minutes: Token expiration time in minutes
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: The plain text password
            
        Returns:
            The hashed password
        """
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            plain_password: The plain text password
            hashed_password: The hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: The data to encode in the token
            expires_delta: Custom expiration time
            
        Returns:
            The encoded JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        if isinstance(encoded_jwt, bytes):
            encoded_jwt = encoded_jwt.decode()
        
        logger.debug(f"Created access token for user {data.get('sub', 'unknown')}")
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: The JWT token to verify
            
        Returns:
            The decoded token data or None if invalid
        """
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm]
            )
            logger.debug(f"Token verified for user {payload.get('sub', 'unknown')}")
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    def generate_api_key(self, length: int = 32) -> str:
        """
        Generate a secure API key.
        
        Args:
            length: The length of the API key
            
        Returns:
            A secure random API key
        """
        return secrets.token_urlsafe(length)
    
    def generate_correlation_id(self) -> str:
        """
        Generate a correlation ID for request tracking.
        
        Returns:
            A unique correlation ID
        """
        return secrets.token_urlsafe(16)


class DataEncryption:
    """Data encryption utilities for sensitive information."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize the data encryption.
        
        Args:
            encryption_key: The encryption key (generates new one if None)
        """
        if encryption_key:
            self.key = encryption_key.encode()
        else:
            # Derive a stable key from the application secret to avoid
            # generating a new key on every restart (which would make
            # previously encrypted data unrecoverable).
            # We use an HMAC-based derivation to produce 32 urlsafe bytes.
            # Note: For production, prefer a KDF with a dedicated salt from Vault.
            try:
                import base64
                derived = hmac.new(
                    key=("AURA_DERIVE_KEY".encode()),
                    msg=("default".encode()),
                    digestmod=hashlib.sha256,
                ).digest()
                self.key = base64.urlsafe_b64encode(derived)
            except Exception:
                # Fallback to random key as last resort
                self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: The data to encrypt
            
        Returns:
            The encrypted data as base64 string
        """
        encrypted_data = self.cipher.encrypt(data.encode())
        return encrypted_data.decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: The encrypted data
            
        Returns:
            The decrypted data
        """
        decrypted_data = self.cipher.decrypt(encrypted_data.encode())
        return decrypted_data.decode()
    
    def get_encryption_key(self) -> str:
        """
        Get the encryption key.
        
        Returns:
            The encryption key as base64 string
        """
        return self.key.decode()


class WebhookSecurity:
    """Webhook security utilities for Instagram and other integrations."""
    
    def __init__(self, webhook_secret: str):
        """
        Initialize webhook security.
        
        Args:
            webhook_secret: The webhook secret for verification
        """
        self.webhook_secret = webhook_secret
    
    def verify_instagram_webhook(
        self,
        payload: bytes,
        signature: str,
        algorithm: str = "sha256",
    ) -> bool:
        """
        Verify Instagram webhook signature.
        
        Args:
            payload: The webhook payload
            signature: The signature from the webhook header
            algorithm: The hashing algorithm
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Remove 'sha256=' prefix if present
            if signature.startswith('sha256='):
                signature = signature[7:]
            
            # Create expected signature from raw request bytes
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            is_valid = hmac.compare_digest(signature, expected_signature)
            
            if is_valid:
                logger.debug("Instagram webhook signature verified")
            else:
                logger.warning("Instagram webhook signature verification failed")
            
            return is_valid
        except Exception as e:
            logger.error(f"Webhook verification error: {e}")
            return False
    
    def verify_shopify_webhook(
        self,
        payload: str,
        signature: str,
    ) -> bool:
        """
        Verify Shopify webhook signature.
        
        Args:
            payload: The webhook payload
            signature: The signature from the webhook header
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Create expected signature
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            is_valid = hmac.compare_digest(signature, expected_signature)
            
            if is_valid:
                logger.debug("Shopify webhook signature verified")
            else:
                logger.warning("Shopify webhook signature verification failed")
            
            return is_valid
        except Exception as e:
            logger.error(f"Shopify webhook verification error: {e}")
            return False


class TenantSecurity:
    """Tenant-specific security utilities."""
    
    @staticmethod
    def validate_tenant_access(
        user_tenant_id: UUID,
        requested_tenant_id: UUID,
    ) -> bool:
        """
        Validate that a user can access a specific tenant's data.
        
        Args:
            user_tenant_id: The user's tenant ID
            requested_tenant_id: The requested tenant ID
            
        Returns:
            True if access is allowed, False otherwise
        """
        is_valid = user_tenant_id == requested_tenant_id
        
        if not is_valid:
            logger.warning(
                f"Tenant access denied: user {user_tenant_id} "
                f"requested access to {requested_tenant_id}"
            )
        
        return is_valid
    
    @staticmethod
    def sanitize_tenant_data(data: Dict[str, Any], tenant_id: UUID) -> Dict[str, Any]:
        """
        Sanitize data to ensure tenant isolation.
        
        Args:
            data: The data to sanitize
            tenant_id: The tenant ID to enforce
            
        Returns:
            The sanitized data
        """
        sanitized_data = data.copy()
        
        # Ensure tenant_id is set correctly
        sanitized_data['tenant_id'] = str(tenant_id)
        
        # Remove any potentially dangerous fields
        dangerous_fields = ['id', 'created_at', 'updated_at']
        for field in dangerous_fields:
            sanitized_data.pop(field, None)
        
        logger.debug(f"Sanitized data for tenant {tenant_id}")
        return sanitized_data
    
    @staticmethod
    def generate_tenant_api_key(tenant_id: UUID) -> str:
        """
        Generate a tenant-specific API key.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            A tenant-specific API key
        """
        # Create a deterministic but secure key based on tenant ID
        tenant_str = str(tenant_id)
        salt = secrets.token_urlsafe(16)
        key_data = f"{tenant_str}:{salt}"
        
        # Hash the key data
        api_key = hashlib.sha256(key_data.encode()).hexdigest()
        
        logger.debug(f"Generated API key for tenant {tenant_id}")
        return api_key


class InputValidator:
    """Input validation utilities for security."""
    
    @staticmethod
    def validate_uuid(uuid_string: str) -> bool:
        """
        Validate if a string is a valid UUID.
        
        Args:
            uuid_string: The string to validate
            
        Returns:
            True if valid UUID, False otherwise
        """
        try:
            UUID(uuid_string)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate if a string is a valid email address.
        
        Args:
            email: The email string to validate
            
        Returns:
            True if valid email, False otherwise
        """
        import re
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def sanitize_string(input_string: str, max_length: int = 1000) -> str:
        """
        Sanitize a string input.
        
        Args:
            input_string: The string to sanitize
            max_length: Maximum allowed length
            
        Returns:
            The sanitized string
        """
        if not isinstance(input_string, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = input_string.strip()
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
        
        return sanitized
    
    @staticmethod
    def validate_instagram_user_id(user_id: str) -> bool:
        """
        Validate Instagram user ID format.
        
        Args:
            user_id: The Instagram user ID
            
        Returns:
            True if valid format, False otherwise
        """
        if not isinstance(user_id, str):
            return False
        
        # Instagram user IDs are typically numeric strings
        return user_id.isdigit() and len(user_id) > 0


# Global security instances
_security_manager: Optional[SecurityManager] = None
_data_encryption: Optional[DataEncryption] = None
_webhook_security: Optional[WebhookSecurity] = None


def initialize_security(
    secret_key: str,
    webhook_secret: str,
    encryption_key: Optional[str] = None,
) -> None:
    """
    Initialize global security instances.
    
    Args:
        secret_key: The secret key for JWT signing
        webhook_secret: The webhook secret
        encryption_key: The encryption key (optional)
    """
    global _security_manager, _data_encryption, _webhook_security
    
    _security_manager = SecurityManager(secret_key)
    # Derive a strong Fernet key if not provided using HKDF from the app secret
    if not encryption_key:
        salt = getenv("ENCRYPTION_SALT", "aura-default-salt").encode()
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"aura-data-encryption",
            backend=default_backend(),
        )
        derived = hkdf.derive(secret_key.encode())
        encryption_key = base64.urlsafe_b64encode(derived).decode()
    _data_encryption = DataEncryption(encryption_key)
    _webhook_security = WebhookSecurity(webhook_secret)
    
    logger.info("Security utilities initialized")


def get_security_manager() -> SecurityManager:
    """
    Get the global security manager.
    
    Returns:
        The security manager instance
        
    Raises:
        RuntimeError: If security manager is not initialized
    """
    if _security_manager is None:
        raise RuntimeError("Security manager not initialized")
    return _security_manager


def get_data_encryption() -> DataEncryption:
    """
    Get the global data encryption instance.
    
    Returns:
        The data encryption instance
        
    Raises:
        RuntimeError: If data encryption is not initialized
    """
    if _data_encryption is None:
        raise RuntimeError("Data encryption not initialized")
    return _data_encryption


def get_webhook_security() -> WebhookSecurity:
    """
    Get the global webhook security instance.
    
    Returns:
        The webhook security instance
        
    Raises:
        RuntimeError: If webhook security is not initialized
    """
    if _webhook_security is None:
        raise RuntimeError("Webhook security not initialized")
    return _webhook_security
