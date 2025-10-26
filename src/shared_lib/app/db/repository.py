"""
Aura Platform - Repository Pattern with Tenant Isolation

Implements the Repository pattern with automatic tenant isolation enforcement.
This module provides secure data access patterns that prevent cross-tenant
data leakage by automatically applying tenant_id filters to all database
operations for tenant-aware models.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

from shared_lib.app.schemas.models import Base, TenantMixin

logger = logging.getLogger(__name__)

# Type variables for generic operations
T = TypeVar("T", bound=Base)


class TenantScopeError(Exception):
    """Raised when attempting to access tenant-scoped data without proper tenant context.
    
    This exception is raised when repository operations are attempted on
    tenant-aware models without providing a valid tenant_id, preventing
    potential data leakage between tenants.
    """
    pass


class BaseRepository:
    """Base repository with automatic tenant isolation."""
    
    def __init__(self, session: AsyncSession, tenant_id: UUID):
        """
        Initialize repository with tenant context.
        
        Args:
            session: Async database session
            tenant_id: Tenant ID for data isolation
            
        Raises:
            TenantScopeError: If tenant_id is None or invalid
        """
        if not tenant_id:
            raise TenantScopeError("Repository must be initialized with a valid tenant_id")
        
        self.session = session
        self.tenant_id = tenant_id
        logger.debug(f"Repository initialized for tenant: {tenant_id}")
    
    def _ensure_tenant_scope(self, model_class: Type[T]) -> None:
        """
        Ensure the model class has tenant isolation.
        
        Args:
            model_class: The model class to check
            
        Raises:
            TenantScopeError: If model doesn't support tenant isolation
        """
        if not hasattr(model_class, 'tenant_id'):
            raise TenantScopeError(
                f"Model {model_class.__name__} does not support tenant isolation. "
                "It must inherit from TenantMixin."
            )
    
    def _apply_tenant_filter(self, query, model_class: Type[T]):
        """
        Apply tenant filter to query.
        
        Args:
            query: SQLAlchemy query object
            model_class: The model class
            
        Returns:
            Query with tenant filter applied
        """
        self._ensure_tenant_scope(model_class)
        return query.where(model_class.tenant_id == self.tenant_id)
    
    async def create(
        self,
        model_class: Type[T],
        **kwargs: Any,
    ) -> T:
        """
        Create a new record with tenant isolation.
        
        Args:
            model_class: The model class
            **kwargs: Model attributes
            
        Returns:
            The created model instance
            
        Raises:
            TenantScopeError: If model doesn't support tenant isolation
        """
        self._ensure_tenant_scope(model_class)
        
        # Automatically set tenant_id
        kwargs['tenant_id'] = self.tenant_id
        
        instance = model_class(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        
        logger.info(f"Created {model_class.__name__} with ID {instance.id} for tenant {self.tenant_id}")
        return instance
    
    async def get_by_id(
        self,
        model_class: Type[T],
        record_id: UUID,
    ) -> Optional[T]:
        """
        Get record by ID with tenant isolation.
        
        Args:
            model_class: The model class
            record_id: Record ID
            
        Returns:
            The record if found, None otherwise
            
        Raises:
            TenantScopeError: If model doesn't support tenant isolation
        """
        self._ensure_tenant_scope(model_class)
        
        query = select(model_class).where(model_class.id == record_id)
        query = self._apply_tenant_filter(query, model_class)
        
        result = await self.session.execute(query)
        instance = result.scalar_one_or_none()
        
        if instance:
            logger.debug(f"Retrieved {model_class.__name__} {record_id} for tenant {self.tenant_id}")
        else:
            logger.debug(f"No {model_class.__name__} {record_id} found for tenant {self.tenant_id}")
        
        return instance
    
    async def get_all(
        self,
        model_class: Type[T],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[T]:
        """
        Get all records with tenant isolation.
        
        Args:
            model_class: The model class
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List of records
            
        Raises:
            TenantScopeError: If model doesn't support tenant isolation
        """
        self._ensure_tenant_scope(model_class)
        
        query = select(model_class)
        query = self._apply_tenant_filter(query, model_class)
        
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        instances = result.scalars().all()
        
        logger.debug(f"Retrieved {len(instances)} {model_class.__name__} records for tenant {self.tenant_id}")
        return list(instances)
    
    async def update(
        self,
        model_class: Type[T],
        record_id: UUID,
        **kwargs: Any,
    ) -> Optional[T]:
        """
        Update record with tenant isolation.
        
        Args:
            model_class: The model class
            record_id: Record ID
            **kwargs: Fields to update
            
        Returns:
            Updated record if found, None otherwise
            
        Raises:
            TenantScopeError: If model doesn't support tenant isolation
        """
        self._ensure_tenant_scope(model_class)
        
        # Prevent tenant_id modification
        if 'tenant_id' in kwargs:
            raise ValueError("Cannot modify tenant_id of existing record")
        
        query = update(model_class).where(model_class.id == record_id)
        query = self._apply_tenant_filter(query, model_class)
        
        result = await self.session.execute(query.values(**kwargs))
        
        if result.rowcount > 0:
            await self.session.flush()
            # Return updated record
            return await self.get_by_id(model_class, record_id)
        else:
            logger.warning(f"No {model_class.__name__} {record_id} found for tenant {self.tenant_id} to update")
            return None
    
    async def delete(
        self,
        model_class: Type[T],
        record_id: UUID,
    ) -> bool:
        """
        Delete record with tenant isolation.
        
        Args:
            model_class: The model class
            record_id: Record ID
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            TenantScopeError: If model doesn't support tenant isolation
        """
        self._ensure_tenant_scope(model_class)
        
        query = delete(model_class).where(model_class.id == record_id)
        query = self._apply_tenant_filter(query, model_class)
        
        result = await self.session.execute(query)
        
        if result.rowcount > 0:
            logger.info(f"Deleted {model_class.__name__} {record_id} for tenant {self.tenant_id}")
            return True
        else:
            logger.warning(f"No {model_class.__name__} {record_id} found for tenant {self.tenant_id} to delete")
            return False
    
    async def count(
        self,
        model_class: Type[T],
    ) -> int:
        """
        Count records with tenant isolation.
        
        Args:
            model_class: The model class
            
        Returns:
            Number of records
            
        Raises:
            TenantScopeError: If model doesn't support tenant isolation
        """
        self._ensure_tenant_scope(model_class)
        
        query = select(model_class)
        query = self._apply_tenant_filter(query, model_class)
        
        result = await self.session.execute(query)
        count = len(result.scalars().all())
        
        logger.debug(f"Counted {count} {model_class.__name__} records for tenant {self.tenant_id}")
        return count
    
    async def exists(
        self,
        model_class: Type[T],
        record_id: UUID,
    ) -> bool:
        """
        Check if record exists with tenant isolation.
        
        Args:
            model_class: The model class
            record_id: Record ID
            
        Returns:
            True if exists, False otherwise
            
        Raises:
            TenantScopeError: If model doesn't support tenant isolation
        """
        return await self.get_by_id(model_class, record_id) is not None


class TenantAwareRepository(BaseRepository):
    """
    Tenant-aware repository with additional tenant-specific operations.
    This is the main repository class to use throughout the application.
    """
    
    async def get_tenant_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current tenant.
        
        Returns:
            Dictionary with tenant statistics
        """
        from shared_lib.app.schemas.models import Product, Conversation, Message
        
        stats = {
            'tenant_id': str(self.tenant_id),
            'products_count': await self.count(Product),
            'conversations_count': await self.count(Conversation),
            'messages_count': await self.count(Message),
        }
        
        logger.info(f"Retrieved stats for tenant {self.tenant_id}: {stats}")
        return stats
    
    async def get_tenant_products_with_pagination(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Any]:
        """
        Get products for tenant with pagination.
        
        Args:
            limit: Maximum number of products
            offset: Number of products to skip
            
        Returns:
            List of products
        """
        from shared_lib.app.schemas.models import Product
        
        return await self.get_all(Product, limit=limit, offset=offset)
    
    async def search_tenant_products(
        self,
        search_term: str,
        limit: int = 20,
    ) -> List[Any]:
        """
        Search products for tenant by title or description.
        
        Args:
            search_term: Search term
            limit: Maximum number of results
            
        Returns:
            List of matching products
        """
        from shared_lib.app.schemas.models import Product
        
        self._ensure_tenant_scope(Product)
        
        query = select(Product).where(
            Product.title.ilike(f"%{search_term}%") |
            Product.description.ilike(f"%{search_term}%")
        )
        query = self._apply_tenant_filter(query, Product)
        query = query.limit(limit)
        
        result = await self.session.execute(query)
        products = result.scalars().all()
        
        logger.info(f"Found {len(products)} products matching '{search_term}' for tenant {self.tenant_id}")
        return list(products)


# Factory function for creating repositories
def create_repository(session: AsyncSession, tenant_id: UUID) -> TenantAwareRepository:
    """
    Create a tenant-aware repository instance.
    
    Args:
        session: Async database session
        tenant_id: Tenant ID
        
    Returns:
        TenantAwareRepository instance
        
    Raises:
        TenantScopeError: If tenant_id is None or invalid
    """
    return TenantAwareRepository(session, tenant_id)
