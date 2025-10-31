"""
Aura Platform - Transactional Decorator Tests
Tests for transaction management and tenant secrets.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

from shared_lib.app.db.database import transactional, TransactionalContext
from sqlalchemy.ext.asyncio import AsyncSession
from shared_lib.app.utils.secrets import TenantSecretsManager, create_secrets_manager
from shared_lib.app.utils.security import initialize_security, get_data_encryption
import os


class TestTransactionalDecorator:
    """Test transactional decorator functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock async database session."""
        session = AsyncMock()
        session.in_transaction.return_value = False
        session.begin.return_value.__aenter__ = AsyncMock()
        session.begin.return_value.__aexit__ = AsyncMock()
        return session
    
    @pytest.fixture
    def tenant_id(self):
        """Valid tenant ID."""
        return uuid4()
    
    @pytest.mark.asyncio
    async def test_transactional_decorator_success(self, mock_session, tenant_id):
        """Test transactional decorator with successful execution."""
        
        @transactional
        async def test_function(session: AsyncSession, tenant_id: UUID):
            """Test function that should succeed."""
            return {"success": True, "tenant_id": tenant_id}
        
        # Mock successful transaction
        mock_session.begin.return_value.__aenter__.return_value = None
        mock_session.begin.return_value.__aexit__.return_value = None
        
        # Execute function
        result = await test_function(mock_session, tenant_id)
        
        # Verify result
        assert result["success"] is True
        assert result["tenant_id"] == tenant_id
        
        # Verify transaction was started
        mock_session.begin.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transactional_decorator_failure(self, mock_session, tenant_id):
        """Test transactional decorator with exception."""
        
        @transactional
        async def test_function(session: AsyncSession, tenant_id: UUID):
            """Test function that should fail."""
            raise ValueError("Test error")
        
        # Mock transaction with exception handling
        mock_session.begin.return_value.__aenter__.return_value = None
        mock_session.begin.return_value.__aexit__.return_value = None
        
        # Execute function and expect exception
        with pytest.raises(ValueError, match="Test error"):
            await test_function(mock_session, tenant_id)
        
        # Verify transaction was started
        mock_session.begin.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transactional_decorator_nested_transaction(self, mock_session, tenant_id):
        """Test transactional decorator with nested transaction."""
        
        @transactional
        async def outer_function(session: AsyncSession, tenant_id: UUID):
            """Outer function with transaction."""
            return await inner_function(session, tenant_id)
        
        @transactional
        async def inner_function(session: AsyncSession, tenant_id: UUID):
            """Inner function with transaction."""
            return {"nested": True, "tenant_id": tenant_id}
        
        # Mock nested transaction scenario
        mock_session.in_transaction.side_effect = [False, True]  # Outer starts, inner is nested
        
        # Execute function
        result = await outer_function(mock_session, tenant_id)
        
        # Verify result
        assert result["nested"] is True
        assert result["tenant_id"] == tenant_id
        
        # Verify only outer transaction was started
        mock_session.begin.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transactional_decorator_missing_session(self, tenant_id):
        """Test transactional decorator without session parameter."""
        
        @transactional
        async def test_function(tenant_id: UUID):
            """Test function without session parameter."""
            return {"success": True}
        
        # Execute function and expect ValueError
        with pytest.raises(ValueError, match="must have an AsyncSession parameter"):
            await test_function(tenant_id)
    
    @pytest.mark.asyncio
    async def test_transactional_context_success(self, mock_session, tenant_id):
        """Test TransactionalContext with successful execution."""
        
        async def test_operation(session: AsyncSession, tenant_id: UUID):
            """Test operation within transaction context."""
            return {"success": True, "tenant_id": tenant_id}
        
        # Mock successful transaction
        mock_session.begin.return_value.__aenter__.return_value = None
        mock_session.begin.return_value.__aexit__.return_value = None
        
        # Execute with context
        async with TransactionalContext(mock_session) as tx:
            result = await test_operation(mock_session, tenant_id)
        
        # Verify result
        assert result["success"] is True
        assert result["tenant_id"] == tenant_id
        
        # Verify transaction was started
        mock_session.begin.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transactional_context_failure(self, mock_session, tenant_id):
        """Test TransactionalContext with exception."""
        
        async def test_operation(session: AsyncSession, tenant_id: UUID):
            """Test operation that fails."""
            raise ValueError("Test error")
        
        # Mock transaction with exception handling
        mock_session.begin.return_value.__aenter__.return_value = None
        mock_session.begin.return_value.__aexit__.return_value = None
        
        # Execute with context and expect exception
        with pytest.raises(ValueError, match="Test error"):
            async with TransactionalContext(mock_session) as tx:
                await test_operation(mock_session, tenant_id)
        
        # Verify transaction was started
        mock_session.begin.assert_called_once()


class TestTenantSecretsManager:
    """Test TenantSecretsManager functionality."""
    
    @pytest.fixture
    def vault_url(self):
        """Vault URL for testing."""
        return "http://localhost:8200"
    
    @pytest.fixture
    def vault_token(self):
        """Vault token for testing."""
        return "test-token"
    
    @pytest.fixture
    def tenant_id(self):
        """Valid tenant ID."""
        return uuid4()
    
    @pytest.fixture
    def secrets_manager(self, vault_url, vault_token):
        """Secrets manager instance."""
        with patch('shared_lib.app.utils.secrets.hvac'):
            return TenantSecretsManager(vault_url, vault_token)
    
    @pytest.mark.asyncio
    async def test_get_secret_success(self, secrets_manager, tenant_id):
        """Test successful secret retrieval."""
        secret_key = "test_secret"
        secret_value = "test_value"
        
        # Mock Vault response
        mock_response = {
            "data": {
                "data": {
                    secret_key: secret_value
                }
            }
        }
        
        secrets_manager.client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        # Test secret retrieval
        result = await secrets_manager.get_secret(tenant_id, secret_key)
        
        # Verify result
        assert result == secret_value
        
        # Verify Vault was called correctly
        expected_path = f"secret/data/aura/tenants/{tenant_id}"
        secrets_manager.client.secrets.kv.v2.read_secret_version.assert_called_once_with(
            path=expected_path,
            mount_point="secret",
        )
    
    @pytest.mark.asyncio
    async def test_get_secret_not_found(self, secrets_manager, tenant_id):
        """Test secret retrieval when secret doesn't exist."""
        secret_key = "nonexistent_secret"
        
        # Mock Vault response with no data
        mock_response = {
            "data": {
                "data": {}
            }
        }
        
        secrets_manager.client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        # Test secret retrieval
        result = await secrets_manager.get_secret(tenant_id, secret_key)
        
        # Verify result
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_all_secrets(self, secrets_manager, tenant_id):
        """Test retrieval of all secrets."""
        secrets = {
            "secret1": "value1",
            "secret2": "value2",
            "secret3": "value3",
        }
        
        # Mock Vault response
        mock_response = {
            "data": {
                "data": secrets
            }
        }
        
        secrets_manager.client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        # Test secret retrieval
        result = await secrets_manager.get_all_secrets(tenant_id)
        
        # Verify result
        assert result == secrets
        assert len(result) == 3
    
    @pytest.mark.asyncio
    async def test_set_secret(self, secrets_manager, tenant_id):
        """Test secret setting."""
        secret_key = "new_secret"
        secret_value = "new_value"
        
        # Mock existing secrets
        existing_secrets = {"existing": "value"}
        
        # Mock Vault responses
        secrets_manager.client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": existing_secrets}
        }
        secrets_manager.client.secrets.kv.v2.create_or_update_secret.return_value = {}
        
        # Test secret setting
        result = await secrets_manager.set_secret(tenant_id, secret_key, secret_value)
        
        # Verify result
        assert result is True
        
        # Verify Vault was called correctly
        expected_path = f"secret/data/aura/tenants/{tenant_id}"
        secrets_manager.client.secrets.kv.v2.create_or_update_secret.assert_called_once_with(
            path=expected_path,
            secret={**existing_secrets, secret_key: secret_value},
            mount_point="secret",
        )
    
    @pytest.mark.asyncio
    async def test_delete_secret(self, secrets_manager, tenant_id):
        """Test secret deletion."""
        secret_key = "to_delete"
        
        # Mock existing secrets
        existing_secrets = {
            "keep": "value",
            secret_key: "delete_me",
        }
        
        # Mock Vault responses
        secrets_manager.client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": existing_secrets}
        }
        secrets_manager.client.secrets.kv.v2.create_or_update_secret.return_value = {}
        
        # Test secret deletion
        result = await secrets_manager.delete_secret(tenant_id, secret_key)
        
        # Verify result
        assert result is True
        
        # Verify Vault was called with updated secrets
        expected_secrets = {"keep": "value"}
        secrets_manager.client.secrets.kv.v2.create_or_update_secret.assert_called_once_with(
            path=f"secret/data/aura/tenants/{tenant_id}",
            secret=expected_secrets,
            mount_point="secret",
        )
    
    @pytest.mark.asyncio
    async def test_health_check(self, secrets_manager):
        """Test health check functionality."""
        # Mock Vault health response
        mock_health = {"initialized": True, "sealed": False}
        secrets_manager.client.sys.read_health_status.return_value = mock_health
        secrets_manager.client.is_authenticated.return_value = True
        
        # Test health check
        result = await secrets_manager.health_check()
        
        # Verify result
        assert result["status"] == "healthy"
        assert result["vault_health"] == mock_health
        assert result["authenticated"] is True
    
    def test_create_secrets_manager_factory(self, vault_url, vault_token):
        """Test secrets manager factory function."""
        with patch('shared_lib.app.utils.secrets.hvac'):
            manager = create_secrets_manager(vault_url, vault_token)
            
            assert isinstance(manager, TenantSecretsManager)
            assert manager.vault_url == vault_url
            assert manager.vault_token == vault_token


class TestIntegration:
    """Integration tests for transactional operations with secrets."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock async database session."""
        session = AsyncMock()
        session.in_transaction.return_value = False
        session.begin.return_value.__aenter__ = AsyncMock()
        session.begin.return_value.__aexit__ = AsyncMock()
        return session
    
    @pytest.fixture
    def tenant_id(self):
        """Valid tenant ID."""
        return uuid4()
    
    @pytest.mark.asyncio
    async def test_transactional_with_secrets_integration(self, mock_session, tenant_id):
        """Test transactional operation that uses secrets."""
        
        @transactional
        async def create_product_with_secrets(session: AsyncSession, tenant_id: UUID):
            """Create product using tenant secrets."""
            # This would typically use TenantSecretsManager
            # For testing, we'll simulate the operation
            return {
                "success": True,
                "tenant_id": tenant_id,
                "used_secrets": True,
            }
        
        # Mock successful transaction
        mock_session.begin.return_value.__aenter__.return_value = None
        mock_session.begin.return_value.__aexit__.return_value = None
        
        # Execute function
        result = await create_product_with_secrets(mock_session, tenant_id)
        
        # Verify result
        assert result["success"] is True
        assert result["tenant_id"] == tenant_id
        assert result["used_secrets"] is True
        
        # Verify transaction was started
        mock_session.begin.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_security_hkdf_derivation_roundtrip(monkeypatch):
    """Verify HKDF-derived encryption key produces stable encrypt/decrypt roundtrip."""
    # Fixed salt for deterministic derivation in test
    monkeypatch.setenv("ENCRYPTION_SALT", "unit-test-salt")
    # Initialize without explicit encryption key
    initialize_security(secret_key="root-secret", webhook_secret="whsec")
    enc = get_data_encryption()
    plaintext = "sensitive-data"
    ciphertext = enc.encrypt_data(plaintext)
    assert ciphertext and isinstance(ciphertext, str)
    decrypted = enc.decrypt_data(ciphertext)
    assert decrypted == plaintext

if __name__ == "__main__":
    pytest.main([__file__])
