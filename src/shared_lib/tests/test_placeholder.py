"""
Aura Platform - Shared Library Tests
Phase 1 Implementation - Unit Test Suite
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from uuid import UUID, uuid4

from shared_lib.app.schemas.events import (
    EventType,
    InstagramDirectMessageReceived,
    AIQueryProcessed,
    create_event,
    get_event_class,
)
from shared_lib.app.schemas.models import Tenant, Product, InstagramUser
from shared_lib.app.utils.security import (
    SecurityManager,
    DataEncryption,
    WebhookSecurity,
    TenantSecurity,
    InputValidator,
)


class TestEventSchemas:
    """Test event schema validation and functionality."""
    
    def test_instagram_direct_message_received_validation(self) -> None:
        """Test Instagram direct message event validation."""
        tenant_id = uuid4()
        
        # Valid event
        event = InstagramDirectMessageReceived(
            tenant_id=tenant_id,
            instagram_user_id="123456789",
            instagram_username="test_user",
            message_id="msg_123",
            message_text="Hello, world!",
            source_service="api_gateway",
        )
        
        assert event.event_type == EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED
        assert event.tenant_id == tenant_id
        assert event.instagram_user_id == "123456789"
        assert event.message_text == "Hello, world!"
    
    def test_event_with_media_urls(self) -> None:
        """Test event with media URLs."""
        tenant_id = uuid4()
        
        event = InstagramDirectMessageReceived(
            tenant_id=tenant_id,
            instagram_user_id="123456789",
            instagram_username="test_user",
            message_id="msg_123",
            media_urls=["https://example.com/image.jpg"],
            source_service="api_gateway",
        )
        
        assert event.media_urls == ["https://example.com/image.jpg"]
        assert event.message_text is None
    
    def test_event_validation_error(self) -> None:
        """Test event validation with missing required fields."""
        tenant_id = uuid4()
        
        with pytest.raises(ValueError):
            InstagramDirectMessageReceived(
                tenant_id=tenant_id,
                instagram_user_id="123456789",
                instagram_username="test_user",
                message_id="msg_123",
                # Missing both message_text and media_urls
                source_service="api_gateway",
            )
    
    def test_create_event_function(self) -> None:
        """Test create_event function."""
        tenant_id = uuid4()
        
        event = create_event(
            EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED,
            tenant_id=tenant_id,
            instagram_user_id="123456789",
            instagram_username="test_user",
            message_id="msg_123",
            message_text="Test message",
            source_service="api_gateway",
        )
        
        assert isinstance(event, InstagramDirectMessageReceived)
        assert event.event_type == EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED
    
    def test_get_event_class_function(self) -> None:
        """Test get_event_class function."""
        event_class = get_event_class(EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED)
        assert event_class == InstagramDirectMessageReceived
        
        with pytest.raises(ValueError):
            get_event_class("invalid.event.type")


class TestSecurityUtilities:
    """Test security utility functions."""
    
    def test_security_manager_password_hashing(self) -> None:
        """Test password hashing and verification."""
        security_manager = SecurityManager("test_secret_key")
        
        password = "test_password_123"
        hashed = security_manager.hash_password(password)
        
        assert hashed != password
        assert security_manager.verify_password(password, hashed)
        assert not security_manager.verify_password("wrong_password", hashed)
    
    def test_security_manager_token_creation(self) -> None:
        """Test JWT token creation and verification."""
        security_manager = SecurityManager("test_secret_key")
        
        data = {"sub": "test_user", "tenant_id": str(uuid4())}
        token = security_manager.create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token
        payload = security_manager.verify_token(token)
        assert payload is not None
        assert payload["sub"] == "test_user"
    
    def test_data_encryption(self) -> None:
        """Test data encryption and decryption."""
        encryption = DataEncryption()
        
        original_data = "sensitive_data_123"
        encrypted = encryption.encrypt_data(original_data)
        decrypted = encryption.decrypt_data(encrypted)
        
        assert encrypted != original_data
        assert decrypted == original_data
    
    def test_webhook_security_instagram(self) -> None:
        """Test Instagram webhook signature verification."""
        webhook_security = WebhookSecurity("test_webhook_secret")
        
        payload = '{"test": "data"}'
        signature = "sha256=test_signature"
        
        # This will fail because we're using a test signature
        result = webhook_security.verify_instagram_webhook(payload, signature)
        assert result is False
    
    def test_tenant_security_validation(self) -> None:
        """Test tenant access validation."""
        tenant_id_1 = uuid4()
        tenant_id_2 = uuid4()
        
        # Same tenant should be allowed
        assert TenantSecurity.validate_tenant_access(tenant_id_1, tenant_id_1)
        
        # Different tenant should be denied
        assert not TenantSecurity.validate_tenant_access(tenant_id_1, tenant_id_2)
    
    def test_input_validator_uuid(self) -> None:
        """Test UUID validation."""
        valid_uuid = str(uuid4())
        invalid_uuid = "not-a-uuid"
        
        assert InputValidator.validate_uuid(valid_uuid)
        assert not InputValidator.validate_uuid(invalid_uuid)
    
    def test_input_validator_email(self) -> None:
        """Test email validation."""
        valid_email = "test@example.com"
        invalid_email = "not-an-email"
        
        assert InputValidator.validate_email(valid_email)
        assert not InputValidator.validate_email(invalid_email)
    
    def test_input_validator_string_sanitization(self) -> None:
        """Test string sanitization."""
        dirty_string = "  test\nstring\twith\0null  "
        sanitized = InputValidator.sanitize_string(dirty_string)
        
        assert sanitized == "teststringwithnull"
        assert len(sanitized) <= 1000


@pytest.mark.unit
class TestDatabaseModels:
    """Test database model definitions."""
    
    def test_tenant_model_creation(self) -> None:
        """Test Tenant model creation."""
        tenant = Tenant(
            name="Test Tenant",
            domain="test.example.com",
        )
        
        assert tenant.name == "Test Tenant"
        assert tenant.domain == "test.example.com"
        assert tenant.is_active is True
        assert isinstance(tenant.id, UUID)
    
    def test_product_model_creation(self) -> None:
        """Test Product model creation."""
        tenant_id = uuid4()
        
        product = Product(
            tenant_id=tenant_id,
            external_id="prod_123",
            external_platform="shopify",
            title="Test Product",
            price=99.99,
            currency="USD",
        )
        
        assert product.tenant_id == tenant_id
        assert product.external_id == "prod_123"
        assert product.title == "Test Product"
        assert product.price == 99.99
        assert product.is_available is True
    
    def test_instagram_user_model_creation(self) -> None:
        """Test InstagramUser model creation."""
        tenant_id = uuid4()
        
        user = InstagramUser(
            tenant_id=tenant_id,
            instagram_user_id="123456789",
            username="test_user",
            full_name="Test User",
        )
        
        assert user.tenant_id == tenant_id
        assert user.instagram_user_id == "123456789"
        assert user.username == "test_user"
        assert user.is_business_user is False
        assert user.total_messages == 0


@pytest.mark.integration
class TestSharedLibIntegration:
    """Integration tests for Shared Library."""
    
    def test_event_serialization(self) -> None:
        """Test event serialization and deserialization."""
        tenant_id = uuid4()
        
        # Create event
        event = InstagramDirectMessageReceived(
            tenant_id=tenant_id,
            instagram_user_id="123456789",
            instagram_username="test_user",
            message_id="msg_123",
            message_text="Test message",
            source_service="api_gateway",
        )
        
        # Serialize to JSON
        json_data = event.json()
        assert isinstance(json_data, str)
        
        # Deserialize from JSON
        event_dict = event.dict()
        assert isinstance(event_dict, dict)
        assert event_dict["event_type"] == EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED
    
    def test_security_integration(self) -> None:
        """Test security utilities integration."""
        # Test password hashing with security manager
        security_manager = SecurityManager("test_secret")
        password = "test_password"
        hashed = security_manager.hash_password(password)
        
        # Test token creation
        token = security_manager.create_access_token({"sub": "test_user"})
        
        # Test token verification
        payload = security_manager.verify_token(token)
        assert payload is not None
        assert payload["sub"] == "test_user"


@pytest.mark.e2e
class TestSharedLibE2E:
    """End-to-end tests for Shared Library."""
    
    def test_complete_event_flow(self) -> None:
        """Test complete event creation and processing flow."""
        tenant_id = uuid4()
        
        # Create event
        event = create_event(
            EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED,
            tenant_id=tenant_id,
            instagram_user_id="123456789",
            instagram_username="test_user",
            message_id="msg_123",
            message_text="Hello, world!",
            source_service="api_gateway",
        )
        
        # Validate event
        assert event.event_type == EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED
        assert event.tenant_id == tenant_id
        
        # Serialize event
        json_data = event.json()
        assert isinstance(json_data, str)
        
        # Test security
        security_manager = SecurityManager("test_secret")
        token = security_manager.create_access_token({
            "sub": "test_user",
            "tenant_id": str(tenant_id)
        })
        
        payload = security_manager.verify_token(token)
        assert payload is not None
        assert payload["tenant_id"] == str(tenant_id)
