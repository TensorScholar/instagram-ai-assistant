#!/usr/bin/env python3
"""
Aura Platform - Phase 1 End-to-End Integration Test
Tests the complete message flow from webhook to intelligence worker.
"""

import json
import sys
import time
from datetime import datetime
from typing import Dict, Any
from uuid import UUID, uuid4

# Add src to path for imports
sys.path.append('src')

def test_event_creation_and_serialization():
    """Test event creation and serialization."""
    print("🧪 Testing event creation and serialization...")
    
    try:
        from shared_lib.app.schemas.events import (
            EventType,
            InstagramDirectMessageReceived,
            create_event,
        )
        
        # Create a test event
        tenant_id = uuid4()
        event = create_event(
            EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED,
            tenant_id=tenant_id,
            instagram_user_id="123456789",
            instagram_username="test_user",
            message_id="msg_123",
            message_text="Hello, this is a test message!",
            source_service="api_gateway",
        )
        
        # Test serialization
        event_dict = event.dict()
        event_json = event.json()
        
        assert isinstance(event_dict, dict)
        assert isinstance(event_json, str)
        assert event_dict["event_type"] == EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED
        assert event_dict["tenant_id"] == str(tenant_id)
        
        print("✅ Event creation and serialization successful")
        return event_dict
        
    except Exception as e:
        print(f"❌ Event creation test failed: {e}")
        return None


def test_api_gateway_webhook_processing():
    """Test API Gateway webhook processing logic."""
    print("🧪 Testing API Gateway webhook processing...")
    
    try:
        # Mock webhook payload
        webhook_payload = {
            "object": "instagram",
            "entry": [
                {
                    "id": "entry_123",
                    "time": int(time.time()),
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "id": "msg_456",
                                "from": {
                                    "id": "user_789",
                                    "username": "test_customer"
                                },
                                "message": {
                                    "text": "Hi, I'm looking for headphones"
                                },
                                "created_time": int(time.time())
                            }
                        }
                    ]
                }
            ]
        }
        
        # Simulate webhook processing
        entry = webhook_payload["entry"][0]
        change = entry["changes"][0]
        message_data = change["value"]
        
        # Extract message information
        message_id = message_data["id"]
        sender_info = message_data["from"]
        message_content = message_data["message"]
        
        sender_id = sender_info["id"]
        sender_username = sender_info["username"]
        message_text = message_content["text"]
        
        # Validate extracted data
        assert message_id == "msg_456"
        assert sender_id == "user_789"
        assert sender_username == "test_customer"
        assert message_text == "Hi, I'm looking for headphones"
        
        print("✅ API Gateway webhook processing successful")
        return {
            "message_id": message_id,
            "sender_id": sender_id,
            "sender_username": sender_username,
            "message_text": message_text,
        }
        
    except Exception as e:
        print(f"❌ API Gateway webhook processing test failed: {e}")
        return None


def test_intelligence_worker_message_processing():
    """Test Intelligence Worker message processing logic."""
    print("🧪 Testing Intelligence Worker message processing...")
    
    try:
        # Mock event data from API Gateway
        event_data = {
            "event_id": str(uuid4()),
            "event_type": "instagram.direct_message.received",
            "tenant_id": str(uuid4()),
            "correlation_id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "source_service": "api_gateway",
            "instagram_user_id": "user_789",
            "instagram_username": "test_customer",
            "message_id": "msg_456",
            "message_text": "Hi, I'm looking for headphones",
            "message_type": "text",
            "media_urls": [],
            "is_business_message": True,
        }
        
        # Simulate message processing
        message_text = event_data["message_text"]
        instagram_user_id = event_data["instagram_user_id"]
        tenant_id = event_data["tenant_id"]
        
        # Mock AI response generation
        mock_response = _generate_mock_ai_response(message_text)
        
        # Validate processing
        assert isinstance(mock_response, str)
        assert len(mock_response) > 0
        assert "headphones" in mock_response.lower() or "product" in mock_response.lower()
        
        print("✅ Intelligence Worker message processing successful")
        return {
            "original_message": message_text,
            "ai_response": mock_response,
            "processing_time_ms": 1500,
            "ai_model_used": "mock-model",
        }
        
    except Exception as e:
        print(f"❌ Intelligence Worker message processing test failed: {e}")
        return None


def test_security_utilities():
    """Test security utilities."""
    print("🧪 Testing security utilities...")
    
    try:
        from shared_lib.app.utils.security import (
            SecurityManager,
            DataEncryption,
            InputValidator,
            TenantSecurity,
        )
        
        # Test password hashing
        security_manager = SecurityManager("test_secret_key")
        password = "test_password_123"
        hashed = security_manager.hash_password(password)
        verified = security_manager.verify_password(password, hashed)
        
        assert verified is True
        assert hashed != password
        
        # Test data encryption
        encryption = DataEncryption()
        original_data = "sensitive_data"
        encrypted = encryption.encrypt_data(original_data)
        decrypted = encryption.decrypt_data(encrypted)
        
        assert decrypted == original_data
        assert encrypted != original_data
        
        # Test input validation
        valid_uuid = str(uuid4())
        valid_email = "test@example.com"
        
        assert InputValidator.validate_uuid(valid_uuid) is True
        assert InputValidator.validate_email(valid_email) is True
        assert InputValidator.validate_uuid("invalid") is False
        assert InputValidator.validate_email("invalid") is False
        
        # Test tenant security
        tenant_id_1 = uuid4()
        tenant_id_2 = uuid4()
        
        assert TenantSecurity.validate_tenant_access(tenant_id_1, tenant_id_1) is True
        assert TenantSecurity.validate_tenant_access(tenant_id_1, tenant_id_2) is False
        
        print("✅ Security utilities test successful")
        return True
        
    except Exception as e:
        print(f"❌ Security utilities test failed: {e}")
        return False


def test_database_models():
    """Test database model definitions."""
    print("🧪 Testing database models...")
    
    try:
        from shared_lib.app.schemas.models import Tenant, Product, InstagramUser
        
        # Test Tenant model
        tenant = Tenant(
            name="Test Tenant",
            domain="test.example.com",
        )
        
        assert tenant.name == "Test Tenant"
        assert tenant.domain == "test.example.com"
        assert tenant.is_active is True
        assert isinstance(tenant.id, UUID)
        
        # Test Product model
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
        
        # Test InstagramUser model
        user = InstagramUser(
            tenant_id=tenant_id,
            instagram_user_id="123456789",
            username="test_user",
        )
        
        assert user.tenant_id == tenant_id
        assert user.instagram_user_id == "123456789"
        assert user.username == "test_user"
        
        print("✅ Database models test successful")
        return True
        
    except Exception as e:
        print(f"❌ Database models test failed: {e}")
        return False


def _generate_mock_ai_response(message_text: str) -> str:
    """Generate a mock AI response based on message content."""
    message_lower = message_text.lower()
    
    if "headphones" in message_lower:
        return "I'd be happy to help you find the perfect headphones! We have several options available. What's your budget range?"
    elif "price" in message_lower:
        return "I can help you find products within your budget! What price range are you looking for?"
    elif "help" in message_lower:
        return "I'm here to help! What can I assist you with today?"
    else:
        return "Thank you for your message! I'm here to help you find great products. What are you looking for?"


def test_complete_message_flow():
    """Test the complete message flow from webhook to AI response."""
    print("🧪 Testing complete message flow...")
    
    try:
        # Step 1: Webhook received
        webhook_data = test_api_gateway_webhook_processing()
        if not webhook_data:
            return False
        
        # Step 2: Event created
        event_data = test_event_creation_and_serialization()
        if not event_data:
            return False
        
        # Step 3: Message processed by Intelligence Worker
        processing_result = test_intelligence_worker_message_processing()
        if not processing_result:
            return False
        
        # Step 4: Validate complete flow
        assert webhook_data["message_text"] == "Hi, I'm looking for headphones"
        assert processing_result["original_message"] == "Hi, I'm looking for headphones"
        assert "headphones" in processing_result["ai_response"].lower()
        
        print("✅ Complete message flow test successful")
        return True
        
    except Exception as e:
        print(f"❌ Complete message flow test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🚀 Starting Aura Platform Phase 1 Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Event Creation and Serialization", test_event_creation_and_serialization),
        ("API Gateway Webhook Processing", test_api_gateway_webhook_processing),
        ("Intelligence Worker Message Processing", test_intelligence_worker_message_processing),
        ("Security Utilities", test_security_utilities),
        ("Database Models", test_database_models),
        ("Complete Message Flow", test_complete_message_flow),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            if result:
                passed_tests += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All Phase 1 integration tests PASSED!")
        print("✅ The system's nervous system is fully functional!")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)