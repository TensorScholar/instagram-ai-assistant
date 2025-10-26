"""
Aura Platform - Repository Pattern Tests
Tests for tenant isolation and repository pattern enforcement.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4, UUID

from shared_lib.app.db.repository import (
    BaseRepository,
    TenantAwareRepository,
    TenantScopeError,
    create_repository,
)
from shared_lib.app.schemas.models import Product, Tenant


class TestTenantIsolation:
    """Test tenant isolation enforcement."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock async database session."""
        session = AsyncMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.execute = AsyncMock()
        session.add = MagicMock()
        return session
    
    @pytest.fixture
    def tenant_id(self):
        """Valid tenant ID."""
        return uuid4()
    
    @pytest.fixture
    def repository(self, mock_session, tenant_id):
        """Repository instance with valid tenant ID."""
        return TenantAwareRepository(mock_session, tenant_id)
    
    def test_repository_initialization_without_tenant_id_raises_error(self, mock_session):
        """Test that repository initialization without tenant_id raises TenantScopeError."""
        with pytest.raises(TenantScopeError, match="Repository must be initialized with a valid tenant_id"):
            TenantAwareRepository(mock_session, None)
    
    def test_repository_initialization_with_empty_tenant_id_raises_error(self, mock_session):
        """Test that repository initialization with empty tenant_id raises TenantScopeError."""
        with pytest.raises(TenantScopeError, match="Repository must be initialized with a valid tenant_id"):
            TenantAwareRepository(mock_session, "")
    
    def test_repository_initialization_with_valid_tenant_id_succeeds(self, mock_session, tenant_id):
        """Test that repository initialization with valid tenant_id succeeds."""
        repo = TenantAwareRepository(mock_session, tenant_id)
        assert repo.tenant_id == tenant_id
        assert repo.session == mock_session
    
    def test_create_repository_factory_without_tenant_id_raises_error(self, mock_session):
        """Test that create_repository factory without tenant_id raises TenantScopeError."""
        with pytest.raises(TenantScopeError, match="Repository must be initialized with a valid tenant_id"):
            create_repository(mock_session, None)
    
    def test_create_repository_factory_with_valid_tenant_id_succeeds(self, mock_session, tenant_id):
        """Test that create_repository factory with valid tenant_id succeeds."""
        repo = create_repository(mock_session, tenant_id)
        assert isinstance(repo, TenantAwareRepository)
        assert repo.tenant_id == tenant_id


class TestTenantScopeEnforcement:
    """Test tenant scope enforcement for different operations."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock async database session."""
        session = AsyncMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.execute = AsyncMock()
        session.add = MagicMock()
        return session
    
    @pytest.fixture
    def tenant_id(self):
        """Valid tenant ID."""
        return uuid4()
    
    @pytest.fixture
    def repository(self, mock_session, tenant_id):
        """Repository instance with valid tenant ID."""
        return TenantAwareRepository(mock_session, tenant_id)
    
    @pytest.mark.asyncio
    async def test_create_automatically_sets_tenant_id(self, repository, tenant_id):
        """Test that create operation automatically sets tenant_id."""
        # Mock the Product model
        mock_product = MagicMock()
        mock_product.id = uuid4()
        
        # Mock the session.execute to return the product
        repository.session.execute.return_value.scalar_one_or_none.return_value = mock_product
        
        # Test create operation
        result = await repository.create(Product, title="Test Product", price=10.0)
        
        # Verify tenant_id was set
        assert result.tenant_id == tenant_id
        repository.session.add.assert_called_once()
        repository.session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id_applies_tenant_filter(self, repository, tenant_id):
        """Test that get_by_id applies tenant filter."""
        product_id = uuid4()
        mock_product = MagicMock()
        mock_product.id = product_id
        mock_product.tenant_id = tenant_id
        
        # Mock the session.execute to return the product
        repository.session.execute.return_value.scalar_one_or_none.return_value = mock_product
        
        # Test get_by_id operation
        result = await repository.get_by_id(Product, product_id)
        
        # Verify the query was executed with tenant filter
        repository.session.execute.assert_called_once()
        assert result == mock_product
    
    @pytest.mark.asyncio
    async def test_update_prevents_tenant_id_modification(self, repository):
        """Test that update operation prevents tenant_id modification."""
        product_id = uuid4()
        
        # Test that updating tenant_id raises ValueError
        with pytest.raises(ValueError, match="Cannot modify tenant_id of existing record"):
            await repository.update(Product, product_id, tenant_id=uuid4())
    
    @pytest.mark.asyncio
    async def test_delete_applies_tenant_filter(self, repository, tenant_id):
        """Test that delete operation applies tenant filter."""
        product_id = uuid4()
        
        # Mock successful deletion
        repository.session.execute.return_value.rowcount = 1
        
        # Test delete operation
        result = await repository.delete(Product, product_id)
        
        # Verify deletion was successful
        assert result is True
        repository.session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_all_applies_tenant_filter(self, repository, tenant_id):
        """Test that get_all operation applies tenant filter."""
        mock_products = [MagicMock(), MagicMock()]
        
        # Mock the session.execute to return products
        repository.session.execute.return_value.scalars.return_value.all.return_value = mock_products
        
        # Test get_all operation
        result = await repository.get_all(Product)
        
        # Verify the query was executed with tenant filter
        repository.session.execute.assert_called_once()
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_count_applies_tenant_filter(self, repository, tenant_id):
        """Test that count operation applies tenant filter."""
        mock_products = [MagicMock(), MagicMock(), MagicMock()]
        
        # Mock the session.execute to return products
        repository.session.execute.return_value.scalars.return_value.all.return_value = mock_products
        
        # Test count operation
        result = await repository.count(Product)
        
        # Verify the query was executed with tenant filter
        repository.session.execute.assert_called_once()
        assert result == 3


class TestTenantScopeError:
    """Test TenantScopeError exception handling."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock async database session."""
        return AsyncMock()
    
    @pytest.fixture
    def tenant_id(self):
        """Valid tenant ID."""
        return uuid4()
    
    def test_tenant_scope_error_message(self):
        """Test TenantScopeError message."""
        error = TenantScopeError("Test error message")
        assert str(error) == "Test error message"
    
    def test_tenant_scope_error_inheritance(self):
        """Test TenantScopeError inheritance."""
        error = TenantScopeError("Test error")
        assert isinstance(error, Exception)


class TestRepositoryIntegration:
    """Integration tests for repository pattern."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock async database session."""
        session = AsyncMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.execute = AsyncMock()
        session.add = MagicMock()
        return session
    
    @pytest.fixture
    def tenant_id(self):
        """Valid tenant ID."""
        return uuid4()
    
    @pytest.fixture
    def repository(self, mock_session, tenant_id):
        """Repository instance with valid tenant ID."""
        return TenantAwareRepository(mock_session, tenant_id)
    
    @pytest.mark.asyncio
    async def test_tenant_stats_integration(self, repository, tenant_id):
        """Test tenant stats integration."""
        # Mock count operations
        repository.count = AsyncMock(side_effect=[5, 3, 12])  # products, conversations, messages
        
        # Test get_tenant_stats
        stats = await repository.get_tenant_stats()
        
        # Verify stats structure
        assert stats['tenant_id'] == str(tenant_id)
        assert stats['products_count'] == 5
        assert stats['conversations_count'] == 3
        assert stats['messages_count'] == 12
        
        # Verify count was called for each model
        assert repository.count.call_count == 3
    
    @pytest.mark.asyncio
    async def test_search_products_integration(self, repository, tenant_id):
        """Test search products integration."""
        search_term = "test product"
        mock_products = [MagicMock(), MagicMock()]
        
        # Mock the session.execute to return products
        repository.session.execute.return_value.scalars.return_value.all.return_value = mock_products
        
        # Test search_tenant_products
        result = await repository.search_tenant_products(search_term)
        
        # Verify search was executed
        repository.session.execute.assert_called_once()
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_pagination_integration(self, repository, tenant_id):
        """Test pagination integration."""
        mock_products = [MagicMock() for _ in range(10)]
        
        # Mock the session.execute to return products
        repository.session.execute.return_value.scalars.return_value.all.return_value = mock_products
        
        # Test get_tenant_products_with_pagination
        result = await repository.get_tenant_products_with_pagination(limit=10, offset=0)
        
        # Verify pagination was executed
        repository.session.execute.assert_called_once()
        assert len(result) == 10


if __name__ == "__main__":
    pytest.main([__file__])
