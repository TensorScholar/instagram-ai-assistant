"""
Aura Platform - Database Utilities
Async database connection and operation utilities with tenant isolation.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.pool import NullPool, QueuePool

from .schemas.models import Base, TenantMixin

logger = logging.getLogger(__name__)

# Type variables for generic operations
T = TypeVar("T", bound=Base)


class DatabaseManager:
    """Database manager for async operations with tenant isolation."""
    
    def __init__(
        self,
        database_url: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """
        Initialize the database manager.
        
        Args:
            database_url: The database URL
            echo: Whether to echo SQL statements
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
        """
        self.database_url = database_url
        self.echo = echo
        
        # Create async engine
        self.engine: AsyncEngine = create_async_engine(
            database_url,
            echo=echo,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    
    async def drop_tables(self) -> None:
        """Drop all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")
    
    @asynccontextmanager
    async def get_session(self):
        """
        Get an async database session.
        
        Yields:
            AsyncSession: The database session
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def close(self) -> None:
        """Close the database engine."""
        await self.engine.dispose()
        logger.info("Database engine closed")


class TenantAwareRepository:
    """Repository pattern with tenant isolation."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the repository.
        
        Args:
            db_manager: The database manager instance
        """
        self.db_manager = db_manager
    
    async def create(
        self,
        model_class: Type[T],
        tenant_id: UUID,
        **kwargs: Any,
    ) -> T:
        """
        Create a new record with tenant isolation.
        
        Args:
            model_class: The model class
            tenant_id: The tenant ID
            **kwargs: Model attributes
            
        Returns:
            The created model instance
        """
        async with self.db_manager.get_session() as session:
            # Ensure tenant_id is set
            if hasattr(model_class, 'tenant_id'):
                kwargs['tenant_id'] = tenant_id
            
            instance = model_class(**kwargs)
            session.add(instance)
            await session.commit()
            await session.refresh(instance)
            
            logger.info(f"Created {model_class.__name__} with ID {instance.id}")
            return instance
    
    async def get_by_id(
        self,
        model_class: Type[T],
        tenant_id: UUID,
        record_id: UUID,
    ) -> Optional[T]:
        """
        Get a record by ID with tenant isolation.
        
        Args:
            model_class: The model class
            tenant_id: The tenant ID
            record_id: The record ID
            
        Returns:
            The model instance or None
        """
        async with self.db_manager.get_session() as session:
            query = select(model_class).where(
                model_class.id == record_id
            )
            
            # Add tenant filter if model has tenant_id
            if hasattr(model_class, 'tenant_id'):
                query = query.where(model_class.tenant_id == tenant_id)
            
            result = await session.execute(query)
            instance = result.scalar_one_or_none()
            
            if instance:
                logger.debug(f"Retrieved {model_class.__name__} with ID {record_id}")
            else:
                logger.debug(f"No {model_class.__name__} found with ID {record_id}")
            
            return instance
    
    async def get_all(
        self,
        model_class: Type[T],
        tenant_id: UUID,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **filters: Any,
    ) -> List[T]:
        """
        Get all records with tenant isolation and optional filters.
        
        Args:
            model_class: The model class
            tenant_id: The tenant ID
            limit: Maximum number of records
            offset: Number of records to skip
            **filters: Additional filters
            
        Returns:
            List of model instances
        """
        async with self.db_manager.get_session() as session:
            query = select(model_class)
            
            # Add tenant filter if model has tenant_id
            if hasattr(model_class, 'tenant_id'):
                query = query.where(model_class.tenant_id == tenant_id)
            
            # Add additional filters
            for field, value in filters.items():
                if hasattr(model_class, field):
                    query = query.where(getattr(model_class, field) == value)
            
            # Add pagination
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            instances = result.scalars().all()
            
            logger.debug(f"Retrieved {len(instances)} {model_class.__name__} records")
            return list(instances)
    
    async def update_by_id(
        self,
        model_class: Type[T],
        tenant_id: UUID,
        record_id: UUID,
        **updates: Any,
    ) -> Optional[T]:
        """
        Update a record by ID with tenant isolation.
        
        Args:
            model_class: The model class
            tenant_id: The tenant ID
            record_id: The record ID
            **updates: Fields to update
            
        Returns:
            The updated model instance or None
        """
        async with self.db_manager.get_session() as session:
            query = update(model_class).where(
                model_class.id == record_id
            )
            
            # Add tenant filter if model has tenant_id
            if hasattr(model_class, 'tenant_id'):
                query = query.where(model_class.tenant_id == tenant_id)
            
            query = query.values(**updates)
            
            result = await session.execute(query)
            await session.commit()
            
            if result.rowcount > 0:
                logger.info(f"Updated {model_class.__name__} with ID {record_id}")
                # Return the updated instance
                return await self.get_by_id(model_class, tenant_id, record_id)
            else:
                logger.warning(f"No {model_class.__name__} found to update with ID {record_id}")
                return None
    
    async def delete_by_id(
        self,
        model_class: Type[T],
        tenant_id: UUID,
        record_id: UUID,
    ) -> bool:
        """
        Delete a record by ID with tenant isolation.
        
        Args:
            model_class: The model class
            tenant_id: The tenant ID
            record_id: The record ID
            
        Returns:
            True if deleted, False if not found
        """
        async with self.db_manager.get_session() as session:
            query = delete(model_class).where(
                model_class.id == record_id
            )
            
            # Add tenant filter if model has tenant_id
            if hasattr(model_class, 'tenant_id'):
                query = query.where(model_class.tenant_id == tenant_id)
            
            result = await session.execute(query)
            await session.commit()
            
            if result.rowcount > 0:
                logger.info(f"Deleted {model_class.__name__} with ID {record_id}")
                return True
            else:
                logger.warning(f"No {model_class.__name__} found to delete with ID {record_id}")
                return False
    
    async def count(
        self,
        model_class: Type[T],
        tenant_id: UUID,
        **filters: Any,
    ) -> int:
        """
        Count records with tenant isolation and optional filters.
        
        Args:
            model_class: The model class
            tenant_id: The tenant ID
            **filters: Additional filters
            
        Returns:
            The count of records
        """
        async with self.db_manager.get_session() as session:
            query = select(model_class)
            
            # Add tenant filter if model has tenant_id
            if hasattr(model_class, 'tenant_id'):
                query = query.where(model_class.tenant_id == tenant_id)
            
            # Add additional filters
            for field, value in filters.items():
                if hasattr(model_class, field):
                    query = query.where(getattr(model_class, field) == value)
            
            result = await session.execute(query)
            count = len(result.scalars().all())
            
            logger.debug(f"Counted {count} {model_class.__name__} records")
            return count


class DatabaseHealthChecker:
    """Database health checker for monitoring."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the health checker.
        
        Args:
            db_manager: The database manager instance
        """
        self.db_manager = db_manager
    
    async def check_connection(self) -> bool:
        """
        Check if the database connection is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            async with self.db_manager.get_session() as session:
                await session.execute(select(1))
                logger.debug("Database connection is healthy")
                return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    async def check_tenant_isolation(self, tenant_id: UUID) -> bool:
        """
        Check if tenant isolation is working correctly.
        
        Args:
            tenant_id: The tenant ID to test
            
        Returns:
            True if tenant isolation is working, False otherwise
        """
        try:
            from .schemas.models import Tenant
            
            async with self.db_manager.get_session() as session:
                # Try to access tenant data
                query = select(Tenant).where(Tenant.id == tenant_id)
                result = await session.execute(query)
                tenant = result.scalar_one_or_none()
                
                if tenant:
                    logger.debug(f"Tenant isolation check passed for tenant {tenant_id}")
                    return True
                else:
                    logger.warning(f"Tenant {tenant_id} not found")
                    return False
        except Exception as e:
            logger.error(f"Tenant isolation check failed: {e}")
            return False


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None
_repository: Optional[TenantAwareRepository] = None
_health_checker: Optional[DatabaseHealthChecker] = None


def initialize_database(
    database_url: str,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> DatabaseManager:
    """
    Initialize the global database manager.
    
    Args:
        database_url: The database URL
        echo: Whether to echo SQL statements
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
        
    Returns:
        The database manager instance
    """
    global _db_manager, _repository, _health_checker
    
    _db_manager = DatabaseManager(
        database_url=database_url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
    )
    
    _repository = TenantAwareRepository(_db_manager)
    _health_checker = DatabaseHealthChecker(_db_manager)
    
    logger.info("Database manager initialized")
    return _db_manager


def get_db_manager() -> DatabaseManager:
    """
    Get the global database manager.
    
    Returns:
        The database manager instance
        
    Raises:
        RuntimeError: If database manager is not initialized
    """
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized")
    return _db_manager


def get_repository() -> TenantAwareRepository:
    """
    Get the global repository instance.
    
    Returns:
        The repository instance
        
    Raises:
        RuntimeError: If repository is not initialized
    """
    if _repository is None:
        raise RuntimeError("Repository not initialized")
    return _repository


def get_health_checker() -> DatabaseHealthChecker:
    """
    Get the global health checker instance.
    
    Returns:
        The health checker instance
        
    Raises:
        RuntimeError: If health checker is not initialized
    """
    if _health_checker is None:
        raise RuntimeError("Health checker not initialized")
    return _health_checker
