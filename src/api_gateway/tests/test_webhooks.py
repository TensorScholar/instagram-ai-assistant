"""
Aura Platform - API Gateway Webhook Tests
Tests for the Instagram webhook endpoint, specifically for signature verification.
"""

import hashlib
import hmac
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api_gateway.app.main import app
from src.shared_lib.app.utils.security import WebhookSecurity, initialize_security

# A known secret for testing purposes
TEST_WEBHOOK_SECRET = "test_secret_key_for_webhooks"

@pytest.fixture(scope="module", autouse=True)
def override_settings():
    """Fixture to override application settings for tests."""
    with patch("src.api_gateway.app.core.config.settings.instagram_app_secret", TEST_WEBHOOK_SECRET):
        initialize_security("a", TEST_WEBHOOK_SECRET, "zYG4uuXZBmhXSS05COPzMLR5KAGVZVbGOmTeCKeXOQA=")
        yield

def generate_signature(payload: bytes) -> str:
    """Generate a mock Instagram webhook signature."""
    return hmac.new(
        TEST_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

def test_handle_instagram_webhook_valid_signature():
    """
    Test the Instagram webhook endpoint with a valid signature.
    This test is expected to FAIL before the fix is applied.
    """
    with TestClient(app) as client:
        payload = {
            "object": "instagram",
            "entry": [{"id": "123", "time": 123, "changes": []}]
        }
        # Note: The raw body might have different spacing than json.dumps
        raw_body = json.dumps(payload).encode()
        signature = generate_signature(raw_body)

        headers = {
            "X-Hub-Signature-256": f"sha256={signature}",
            "Content-Type": "application/json"
        }

        response = client.post(
            "/api/v1/webhooks/instagram",
            content=raw_body,
            headers=headers
        )

        # This will initially fail with a 403, but should be 200 after the fix.
        assert response.status_code == 200
        assert response.json()["status"] == "success"

def test_handle_instagram_webhook_invalid_signature():
    """Test the Instagram webhook endpoint with an invalid signature."""
    with TestClient(app) as client:
        payload = {
            "object": "instagram",
            "entry": [{"id": "123", "time": 123, "changes": []}]
        }
        raw_body = json.dumps(payload).encode()

        headers = {
            "X-Hub-Signature-256": "sha256=invalid_signature",
            "Content-Type": "application/json"
        }

        response = client.post(
            "/api/v1/webhooks/instagram",
            content=raw_body,
            headers=headers
        )

        assert response.status_code == 403
        assert response.json()["detail"] == "Invalid webhook signature"
