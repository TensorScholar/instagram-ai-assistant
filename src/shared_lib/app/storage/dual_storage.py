"""
Dual Storage Implementation Module

This module implements the dual storage system that manages data across
PostgreSQL (for metadata) and Milvus (for vectors) with tenant isolation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_
from sqlalchemy.orm import selectinload

from ..schemas.models import Product, Tenant, EventLog
from ..db.database import get_db_session, BaseRepository
from ..ai.vector_store import MilvusVectorStore
from ..ai.embeddings import EmbeddingGenerator, EmbeddingConfig, EmbeddingModel
from ..utils.security import TenantSecurity
from ..schemas.events import ProductIngested, DataIngestionCompleted

logger = logging.getLogger(__name__)


class StorageOperation(Enum):
    """Storage operation types"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"


@dataclass
class StorageResult:
    """Result of storage operation"""
    success: bool
    operation: StorageOperation
    product_id: str
    tenant_id: str
    database_result: Optional[Dict[str, Any]] = None
    vector_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class StorageMetrics:
    """Metrics for storage operations"""
    total_operations: int
    successful_operations: int
    failed_operations: int
    database_operations: int
    vector_operations: int
    average_processing_time: float
    errors_by_type: Dict[str, int]


class DualStorageManager:
    """
    Manages dual storage across PostgreSQL and Milvus with tenant isolation
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.tenant_security = TenantSecurity()
        self.vector_store = MilvusVectorStore(
            host='localhost',
            port=19530,
            collection_name='product_embeddings'
        )
        self.embedding_generator = None
        self.metrics = StorageMetrics(
            total_operations=0,
            successful_operations=0,
            failed_operations=0,
            database_operations=0,
            vector_operations=0,
            average_processing_time=0.0,
            errors_by_type={}
        )
        
    async def initialize(self) -> None:
        """Initialize storage connections"""
        try:
            # Initialize vector store
            await self.vector_store.initialize()
            
            # Initialize embedding generator
            config = EmbeddingConfig(
                model=EmbeddingModel.SENTENCE_TRANSFORMERS_ALL_MPNET,
                batch_size=50
            )
            self.embedding_generator = EmbeddingGenerator(config)
            
            logger.info(f"Initialized dual storage manager for tenant {self.tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize dual storage manager: {e}")
            raise
            
    async def cleanup(self) -> None:
        """Cleanup storage connections"""
        try:
            await self.vector_store.disconnect()
            logger.info(f"Cleaned up dual storage manager for tenant {self.tenant_id}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    async def store_product(self, product_data: Dict[str, Any], operation: StorageOperation = StorageOperation.UPSERT) -> StorageResult:
        """
        Store product data in both PostgreSQL and Milvus
        
        Args:
            product_data: Product data dictionary
            operation: Storage operation type
            
        Returns:
            Storage result with operation details
        """
        start_time = datetime.now()
        product_id = product_data.get('external_id', '')
        
        try:
            # Validate tenant access
            if not await self.tenant_security.validate_tenant_access(self.tenant_id):
                raise ValueError(f"Invalid tenant access: {self.tenant_id}")
                
            # Ensure tenant_id is set
            product_data['tenant_id'] = self.tenant_id
            
            # Store in PostgreSQL
            db_result = await self._store_in_database(product_data, operation)
            
            # Store vectors in Milvus
            vector_result = await self._store_vectors(product_data, operation)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Update metrics
            self._update_metrics(True, processing_time)
            
            result = StorageResult(
                success=True,
                operation=operation,
                product_id=product_id,
                tenant_id=self.tenant_id,
                database_result=db_result,
                vector_result=vector_result,
                processing_time=processing_time
            )
            
            logger.info(f"Successfully stored product {product_id} for tenant {self.tenant_id}")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.error(f"Failed to store product {product_id} for tenant {self.tenant_id}: {e}")
            
            # Update metrics
            self._update_metrics(False, processing_time, str(e))
            
            return StorageResult(
                success=False,
                operation=operation,
                product_id=product_id,
                tenant_id=self.tenant_id,
                error_message=str(e),
                processing_time=processing_time
            )
            
    async def batch_store_products(self, products: List[Dict[str, Any]], operation: StorageOperation = StorageOperation.UPSERT) -> List[StorageResult]:
        """
        Store multiple products in batch
        
        Args:
            products: List of product data dictionaries
            operation: Storage operation type
            
        Returns:
            List of storage results
        """
        results = []
        
        # Process products in batches to avoid overwhelming the system
        batch_size = 50
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.store_product(product, operation) 
                for product in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    results.append(StorageResult(
                        success=False,
                        operation=operation,
                        product_id='unknown',
                        tenant_id=self.tenant_id,
                        error_message=str(result)
                    ))
                else:
                    results.append(result)
                    
        return results
        
    async def retrieve_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve product data from PostgreSQL
        
        Args:
            product_id: Product external ID
            
        Returns:
            Product data dictionary or None
        """
        try:
            async with get_db_session() as session:
                repo = BaseRepository(session)
                
                product = await repo.get_by_external_id(
                    Product, 
                    product_id, 
                    self.tenant_id
                )
                
                if product:
                    return self._product_to_dict(product)
                    
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve product {product_id}: {e}")
            return None
            
    async def search_similar_products(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar products using vector similarity
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of similar products with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_product_embeddings({
                'title': query,
                'description': query
            })
            
            # Search in vector store
            vector_results = await self.vector_store.search_similar(
                query_embedding, 
                self.tenant_id, 
                limit
            )
            
            # Retrieve full product data from PostgreSQL
            similar_products = []
            for result in vector_results:
                product_id = result['metadata']['product_id']
                product_data = await self.retrieve_product(product_id)
                
                if product_data:
                    product_data['similarity_score'] = result['similarity_score']
                    similar_products.append(product_data)
                    
            return similar_products
            
        except Exception as e:
            logger.error(f"Failed to search similar products: {e}")
            return []
            
    async def delete_product(self, product_id: str) -> StorageResult:
        """
        Delete product from both PostgreSQL and Milvus
        
        Args:
            product_id: Product external ID
            
        Returns:
            Storage result
        """
        start_time = datetime.now()
        
        try:
            # Soft delete in PostgreSQL
            db_result = await self._delete_from_database(product_id)
            
            # Delete vectors from Milvus
            vector_result = await self._delete_vectors(product_id)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            self._update_metrics(True, processing_time)
            
            return StorageResult(
                success=True,
                operation=StorageOperation.DELETE,
                product_id=product_id,
                tenant_id=self.tenant_id,
                database_result=db_result,
                vector_result=vector_result,
                processing_time=processing_time
            )
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.error(f"Failed to delete product {product_id}: {e}")
            self._update_metrics(False, processing_time, str(e))
            
            return StorageResult(
                success=False,
                operation=StorageOperation.DELETE,
                product_id=product_id,
                tenant_id=self.tenant_id,
                error_message=str(e),
                processing_time=processing_time
            )
            
    async def sync_product_data(self, product_id: str) -> StorageResult:
        """
        Synchronize product data between PostgreSQL and Milvus
        
        Args:
            product_id: Product external ID
            
        Returns:
            Sync result
        """
        try:
            # Retrieve from PostgreSQL
            product_data = await self.retrieve_product(product_id)
            
            if not product_data:
                return StorageResult(
                    success=False,
                    operation=StorageOperation.UPDATE,
                    product_id=product_id,
                    tenant_id=self.tenant_id,
                    error_message="Product not found in database"
                )
                
            # Regenerate and store vectors
            vector_result = await self._store_vectors(product_data, StorageOperation.UPDATE)
            
            return StorageResult(
                success=True,
                operation=StorageOperation.UPDATE,
                product_id=product_id,
                tenant_id=self.tenant_id,
                vector_result=vector_result
            )
            
        except Exception as e:
            logger.error(f"Failed to sync product {product_id}: {e}")
            return StorageResult(
                success=False,
                operation=StorageOperation.UPDATE,
                product_id=product_id,
                tenant_id=self.tenant_id,
                error_message=str(e)
            )
            
    async def get_storage_health(self) -> Dict[str, Any]:
        """Get health status of both storage systems"""
        try:
            # Check PostgreSQL connection
            db_healthy = await self._check_database_health()
            
            # Check Milvus connection
            vector_healthy = await self.vector_store.health_check()
            
            # Get metrics
            metrics_dict = {
                'total_operations': self.metrics.total_operations,
                'successful_operations': self.metrics.successful_operations,
                'failed_operations': self.metrics.failed_operations,
                'success_rate': (
                    self.metrics.successful_operations / self.metrics.total_operations * 100
                    if self.metrics.total_operations > 0 else 0
                ),
                'average_processing_time': self.metrics.average_processing_time,
                'errors_by_type': self.metrics.errors_by_type
            }
            
            return {
                'status': 'healthy' if db_healthy and vector_healthy else 'unhealthy',
                'database': db_healthy,
                'vector_store': vector_healthy,
                'metrics': metrics_dict,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
    async def _store_in_database(self, product_data: Dict[str, Any], operation: StorageOperation) -> Dict[str, Any]:
        """Store product data in PostgreSQL"""
        async with get_db_session() as session:
            repo = BaseRepository(session)
            
            if operation == StorageOperation.DELETE:
                # Soft delete
                success = await repo.delete_by_external_id(
                    Product, 
                    product_data['external_id'], 
                    self.tenant_id
                )
                return {'success': success, 'operation': 'delete'}
                
            elif operation == StorageOperation.CREATE:
                # Create new product
                product = Product(**product_data)
                success = await repo.create(product, self.tenant_id)
                return {'success': success, 'operation': 'create'}
                
            elif operation == StorageOperation.UPDATE:
                # Update existing product
                existing_product = await repo.get_by_external_id(
                    Product, 
                    product_data['external_id'], 
                    self.tenant_id
                )
                
                if existing_product:
                    success = await repo.update(existing_product.id, product_data, self.tenant_id)
                    return {'success': success, 'operation': 'update'}
                else:
                    # Create if doesn't exist
                    product = Product(**product_data)
                    success = await repo.create(product, self.tenant_id)
                    return {'success': success, 'operation': 'create'}
                    
            elif operation == StorageOperation.UPSERT:
                # Upsert (update or create)
                existing_product = await repo.get_by_external_id(
                    Product, 
                    product_data['external_id'], 
                    self.tenant_id
                )
                
                if existing_product:
                    success = await repo.update(existing_product.id, product_data, self.tenant_id)
                    return {'success': success, 'operation': 'update'}
                else:
                    product = Product(**product_data)
                    success = await repo.create(product, self.tenant_id)
                    return {'success': success, 'operation': 'create'}
                    
            return {'success': False, 'operation': 'unknown'}
            
    async def _store_vectors(self, product_data: Dict[str, Any], operation: StorageOperation) -> Dict[str, Any]:
        """Store product vectors in Milvus"""
        try:
            if operation == StorageOperation.DELETE:
                # Delete vectors
                success = await self.vector_store.delete_vectors(
                    [f"{self.tenant_id}_{product_data['external_id']}"], 
                    self.tenant_id
                )
                return {'success': success, 'operation': 'delete'}
                
            # Generate embeddings
            embeddings = await self.embedding_generator.generate_product_embeddings(product_data)
            
            # Prepare vector data
            vector_data = {
                'id': f"{self.tenant_id}_{product_data['external_id']}",
                'vector': embeddings,
                'metadata': {
                    'tenant_id': self.tenant_id,
                    'product_id': product_data['external_id'],
                    'title': product_data.get('title', ''),
                    'description': product_data.get('description', ''),
                    'category': product_data.get('product_type', ''),
                    'price': product_data.get('price', '0.00'),
                    'created_at': datetime.now(timezone.utc).isoformat()
                }
            }
            
            # Store in Milvus
            success = await self.vector_store.insert_vectors([vector_data], self.tenant_id)
            
            return {
                'success': success, 
                'operation': operation.value,
                'embedding_dimensions': len(embeddings)
            }
            
        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            return {'success': False, 'operation': operation.value, 'error': str(e)}
            
    async def _delete_from_database(self, product_id: str) -> Dict[str, Any]:
        """Delete product from PostgreSQL"""
        async with get_db_session() as session:
            repo = BaseRepository(session)
            success = await repo.delete_by_external_id(Product, product_id, self.tenant_id)
            return {'success': success, 'operation': 'delete'}
            
    async def _delete_vectors(self, product_id: str) -> Dict[str, Any]:
        """Delete vectors from Milvus"""
        try:
            success = await self.vector_store.delete_vectors(
                [f"{self.tenant_id}_{product_id}"], 
                self.tenant_id
            )
            return {'success': success, 'operation': 'delete'}
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return {'success': False, 'operation': 'delete', 'error': str(e)}
            
    def _product_to_dict(self, product: Product) -> Dict[str, Any]:
        """Convert Product model to dictionary"""
        return {
            'id': product.id,
            'tenant_id': product.tenant_id,
            'external_id': product.external_id,
            'title': product.title,
            'description': product.description,
            'handle': product.handle,
            'vendor': product.vendor,
            'product_type': product.product_type,
            'tags': product.tags,
            'status': product.status,
            'price': product.price,
            'compare_at_price': product.compare_at_price,
            'sku': product.sku,
            'inventory_quantity': product.inventory_quantity,
            'weight': product.weight,
            'weight_unit': product.weight_unit,
            'featured_image': product.featured_image,
            'variants': product.variants,
            'images': product.images,
            'options': product.options,
            'seo': product.seo,
            'inventory': product.inventory,
            'pricing': product.pricing,
            'metadata': product.metadata,
            'created_at': product.created_at.isoformat() if product.created_at else None,
            'updated_at': product.updated_at.isoformat() if product.updated_at else None
        }
        
    async def _check_database_health(self) -> bool:
        """Check PostgreSQL connection health"""
        try:
            async with get_db_session() as session:
                # Simple query to test connection
                result = await session.execute(select(1))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
            
    def _update_metrics(self, success: bool, processing_time: float, error: Optional[str] = None) -> None:
        """Update storage metrics"""
        self.metrics.total_operations += 1
        
        if success:
            self.metrics.successful_operations += 1
        else:
            self.metrics.failed_operations += 1
            
            if error:
                error_type = error.split(':')[0] if ':' in error else 'general'
                self.metrics.errors_by_type[error_type] = self.metrics.errors_by_type.get(error_type, 0) + 1
                
        # Update average processing time
        total_time = self.metrics.average_processing_time * (self.metrics.total_operations - 1)
        self.metrics.average_processing_time = (total_time + processing_time) / self.metrics.total_operations


class StorageSynchronizer:
    """
    Handles synchronization between PostgreSQL and Milvus
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.storage_manager = DualStorageManager(tenant_id)
        
    async def sync_all_products(self) -> Dict[str, Any]:
        """Synchronize all products for the tenant"""
        try:
            await self.storage_manager.initialize()
            
            # Get all products from PostgreSQL
            async with get_db_session() as session:
                repo = BaseRepository(session)
                products = await repo.get_all_by_tenant(Product, self.tenant_id)
                
            sync_results = []
            for product in products:
                product_data = self.storage_manager._product_to_dict(product)
                result = await self.storage_manager.sync_product_data(product_data['external_id'])
                sync_results.append(result)
                
            successful_syncs = sum(1 for r in sync_results if r.success)
            
            return {
                'total_products': len(products),
                'successful_syncs': successful_syncs,
                'failed_syncs': len(products) - successful_syncs,
                'sync_rate': (successful_syncs / len(products) * 100) if products else 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to sync all products: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        finally:
            await self.storage_manager.cleanup()
            
    async def validate_data_consistency(self) -> Dict[str, Any]:
        """Validate data consistency between PostgreSQL and Milvus"""
        try:
            await self.storage_manager.initialize()
            
            # Get all products from PostgreSQL
            async with get_db_session() as session:
                repo = BaseRepository(session)
                db_products = await repo.get_all_by_tenant(Product, self.tenant_id)
                
            db_product_ids = {p.external_id for p in db_products}
            
            # Get all vectors from Milvus
            vector_results = await self.storage_manager.vector_store.get_all_vectors(self.tenant_id)
            vector_product_ids = {
                result['metadata']['product_id'] 
                for result in vector_results
            }
            
            # Find inconsistencies
            missing_in_vectors = db_product_ids - vector_product_ids
            missing_in_database = vector_product_ids - db_product_ids
            
            return {
                'total_db_products': len(db_products),
                'total_vector_products': len(vector_results),
                'missing_in_vectors': list(missing_in_vectors),
                'missing_in_database': list(missing_in_database),
                'consistency_rate': (
                    len(db_product_ids & vector_product_ids) / len(db_product_ids | vector_product_ids) * 100
                    if db_product_ids or vector_product_ids else 100
                ),
                'is_consistent': len(missing_in_vectors) == 0 and len(missing_in_database) == 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to validate data consistency: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        finally:
            await self.storage_manager.cleanup()
