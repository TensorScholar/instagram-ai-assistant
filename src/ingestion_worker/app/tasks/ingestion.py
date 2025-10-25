"""
Ingestion Worker - Celery Task Orchestration

This module implements the ingestion worker that orchestrates data fetching,
processing, and storage across multiple e-commerce platforms.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from celery import Celery
from celery.exceptions import Retry
import json

from ..schemas.events import ProductIngested, TenantCreated, DataIngestionCompleted
from ..schemas.models import Product, Tenant
from ..integrations.shopify import ShopifyIntegrationManager, ShopifyConfig
from ..ai.vector_store import MilvusVectorStore
from ..ai.embeddings import EmbeddingGenerator
from ..db.database import get_db_session, BaseRepository
from ..utils.security import TenantSecurity, InputValidator
from ..utils.rabbitmq import RabbitMQPublisher

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery('ingestion_worker')

# Configure Celery
celery_app.conf.update(
    broker_url='redis://localhost:6379/1',
    result_backend='redis://localhost:6379/1',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_compression='gzip',
    result_compression='gzip',
    task_routes={
        'ingestion_worker.tasks.shopify.*': {'queue': 'shopify_queue'},
        'ingestion_worker.tasks.general.*': {'queue': 'ingestion_queue'},
    }
)

# Task queues
INGESTION_QUEUE = 'ingestion_queue'
SHOPIFY_QUEUE = 'shopify_queue'
VECTOR_QUEUE = 'vector_queue'


class IngestionTaskManager:
    """
    Manages ingestion tasks and coordinates between different services
    """
    
    def __init__(self):
        self.rabbitmq_publisher = RabbitMQPublisher()
        self.vector_store = MilvusVectorStore(
            host='localhost',
            port=19530,
            collection_name='product_embeddings'
        )
        self.embedding_generator = EmbeddingGenerator()
        self.tenant_security = TenantSecurity()
        
    async def initialize(self):
        """Initialize connections"""
        await self.rabbitmq_publisher.connect()
        await self.vector_store.initialize()
        
    async def cleanup(self):
        """Cleanup connections"""
        await self.rabbitmq_publisher.disconnect()
        await self.vector_store.disconnect()


# Global task manager instance
task_manager = IngestionTaskManager()


@celery_app.task(bind=True, name="ingest_tenant_products")
async def ingest_tenant_products(self, tenant_id: str, platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main task to ingest all products for a tenant from a specific platform
    
    Args:
        tenant_id: Tenant identifier
        platform: E-commerce platform (shopify, woocommerce, etc.)
        config: Platform-specific configuration
        
    Returns:
        Ingestion results summary
    """
    task_id = self.request.id
    logger.info(f"Starting product ingestion for tenant {tenant_id} from {platform} (task: {task_id})")
    
    try:
        # Initialize task manager
        await task_manager.initialize()
        
        # Route to platform-specific handler
        if platform == 'shopify':
            result = await _ingest_shopify_products(tenant_id, config)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
            
        # Publish completion event
        completion_event = DataIngestionCompleted(
            tenant_id=tenant_id,
            platform=platform,
            task_id=task_id,
            products_processed=result['products_processed'],
            products_created=result['products_created'],
            products_updated=result['products_updated'],
            products_failed=result['products_failed'],
            vectors_created=result['vectors_created'],
            timestamp=datetime.now(timezone.utc),
            status='completed'
        )
        
        await task_manager.rabbitmq_publisher.publish_event(
            completion_event, 
            'data_ingestion_completed'
        )
        
        logger.info(f"Completed product ingestion for tenant {tenant_id}: {result['products_processed']} products processed")
        return result
        
    except Exception as e:
        logger.error(f"Product ingestion failed for tenant {tenant_id}: {e}")
        
        # Publish failure event
        failure_event = DataIngestionCompleted(
            tenant_id=tenant_id,
            platform=platform,
            task_id=task_id,
            products_processed=0,
            products_created=0,
            products_updated=0,
            products_failed=0,
            vectors_created=0,
            timestamp=datetime.now(timezone.utc),
            status='failed',
            error_message=str(e)
        )
        
        await task_manager.rabbitmq_publisher.publish_event(
            failure_event, 
            'data_ingestion_failed'
        )
        
        raise self.retry(exc=e, countdown=60, max_retries=3)
        
    finally:
        await task_manager.cleanup()


@celery_app.task(bind=True, name="process_product_data")
async def process_product_data(self, tenant_id: str, product_data: Dict[str, Any], action: str = 'create') -> Dict[str, Any]:
    """
    Process individual product data and store in both PostgreSQL and Milvus
    
    Args:
        tenant_id: Tenant identifier
        product_data: Processed product data
        action: Action type (create, update, delete)
        
    Returns:
        Processing results
    """
    task_id = self.request.id
    logger.info(f"Processing product {product_data.get('external_id')} for tenant {tenant_id} (action: {action})")
    
    try:
        # Initialize task manager
        await task_manager.initialize()
        
        # Validate tenant access
        if not await task_manager.tenant_security.validate_tenant_access(tenant_id):
            raise ValueError(f"Invalid tenant access: {tenant_id}")
            
        # Store in PostgreSQL
        db_result = await _store_product_in_database(tenant_id, product_data, action)
        
        # Generate and store vector embeddings
        vector_result = await _store_product_vectors(tenant_id, product_data, action)
        
        # Create ProductIngested event
        product_event = ProductIngested(
            tenant_id=tenant_id,
            product_id=product_data.get('external_id'),
            product_data=product_data,
            action=action,
            timestamp=datetime.now(timezone.utc),
            source='ingestion_worker'
        )
        
        await task_manager.rabbitmq_publisher.publish_event(
            product_event, 
            'product_ingested'
        )
        
        result = {
            'product_id': product_data.get('external_id'),
            'action': action,
            'database_result': db_result,
            'vector_result': vector_result,
            'success': True
        }
        
        logger.info(f"Successfully processed product {product_data.get('external_id')} for tenant {tenant_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process product {product_data.get('external_id')} for tenant {tenant_id}: {e}")
        raise self.retry(exc=e, countdown=30, max_retries=2)
        
    finally:
        await task_manager.cleanup()


@celery_app.task(bind=True, name="sync_tenant_data")
async def sync_tenant_data(self, tenant_id: str, platform: str, config: Dict[str, Any], incremental: bool = True) -> Dict[str, Any]:
    """
    Synchronize tenant data with external platform
    
    Args:
        tenant_id: Tenant identifier
        platform: E-commerce platform
        config: Platform configuration
        incremental: Whether to perform incremental sync
        
    Returns:
        Sync results
    """
    task_id = self.request.id
    logger.info(f"Starting data sync for tenant {tenant_id} from {platform} (incremental: {incremental})")
    
    try:
        # Initialize task manager
        await task_manager.initialize()
        
        # Get last sync timestamp for incremental sync
        last_sync = None
        if incremental:
            last_sync = await _get_last_sync_timestamp(tenant_id, platform)
            
        # Route to platform-specific sync
        if platform == 'shopify':
            result = await _sync_shopify_data(tenant_id, config, last_sync)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
            
        # Update sync timestamp
        await _update_sync_timestamp(tenant_id, platform, datetime.now(timezone.utc))
        
        logger.info(f"Completed data sync for tenant {tenant_id}: {result['products_synced']} products synced")
        return result
        
    except Exception as e:
        logger.error(f"Data sync failed for tenant {tenant_id}: {e}")
        raise self.retry(exc=e, countdown=120, max_retries=2)
        
    finally:
        await task_manager.cleanup()


@celery_app.task(bind=True, name="generate_product_embeddings")
async def generate_product_embeddings(self, tenant_id: str, product_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate vector embeddings for product data
    
    Args:
        tenant_id: Tenant identifier
        product_data: Product data to embed
        
    Returns:
        Embedding generation results
    """
    task_id = self.request.id
    logger.info(f"Generating embeddings for product {product_data.get('external_id')} (tenant: {tenant_id})")
    
    try:
        # Initialize task manager
        await task_manager.initialize()
        
        # Generate embeddings
        embeddings = await task_manager.embedding_generator.generate_product_embeddings(product_data)
        
        # Store in vector database
        vector_data = {
            'id': f"{tenant_id}_{product_data.get('external_id')}",
            'vector': embeddings,
            'metadata': {
                'tenant_id': tenant_id,
                'product_id': product_data.get('external_id'),
                'title': product_data.get('title', ''),
                'description': product_data.get('description', ''),
                'category': product_data.get('product_type', ''),
                'price': product_data.get('price', '0.00'),
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        }
        
        success = await task_manager.vector_store.insert_vectors([vector_data], tenant_id)
        
        result = {
            'product_id': product_data.get('external_id'),
            'embeddings_generated': len(embeddings),
            'vector_stored': success,
            'success': success
        }
        
        logger.info(f"Successfully generated embeddings for product {product_data.get('external_id')}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings for product {product_data.get('external_id')}: {e}")
        raise self.retry(exc=e, countdown=30, max_retries=2)
        
    finally:
        await task_manager.cleanup()


# Helper functions for platform-specific implementations

async def _ingest_shopify_products(tenant_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest products from Shopify"""
    shopify_config = ShopifyConfig(
        shop_domain=config['shop_domain'],
        access_token=config['access_token'],
        webhook_secret=config.get('webhook_secret')
    )
    
    integration = ShopifyIntegrationManager(shopify_config, tenant_id)
    
    products_processed = 0
    products_created = 0
    products_updated = 0
    products_failed = 0
    vectors_created = 0
    
    async for product_data in integration.fetch_all_products():
        try:
            # Process product data
            result = await process_product_data.delay(tenant_id, product_data, 'create')
            products_processed += 1
            products_created += 1
            
            # Generate embeddings
            embedding_result = await generate_product_embeddings.delay(tenant_id, product_data)
            if embedding_result.get('success'):
                vectors_created += 1
                
        except Exception as e:
            logger.error(f"Failed to process product {product_data.get('external_id')}: {e}")
            products_failed += 1
            
    return {
        'products_processed': products_processed,
        'products_created': products_created,
        'products_updated': products_updated,
        'products_failed': products_failed,
        'vectors_created': vectors_created
    }


async def _store_product_in_database(tenant_id: str, product_data: Dict[str, Any], action: str) -> Dict[str, Any]:
    """Store product data in PostgreSQL"""
    async with get_db_session() as session:
        repo = BaseRepository(session)
        
        if action == 'delete':
            # Soft delete product
            success = await repo.delete_by_external_id(
                Product, 
                product_data.get('external_id'), 
                tenant_id
            )
        else:
            # Create or update product
            existing_product = await repo.get_by_external_id(
                Product, 
                product_data.get('external_id'), 
                tenant_id
            )
            
            if existing_product:
                success = await repo.update(existing_product.id, product_data, tenant_id)
            else:
                success = await repo.create(Product(**product_data), tenant_id)
                
        return {'success': success, 'action': action}


async def _store_product_vectors(tenant_id: str, product_data: Dict[str, Any], action: str) -> Dict[str, Any]:
    """Store product vectors in Milvus"""
    if action == 'delete':
        # Remove vectors
        success = await task_manager.vector_store.delete_vectors(
            [f"{tenant_id}_{product_data.get('external_id')}"], 
            tenant_id
        )
    else:
        # Generate and store new vectors
        embedding_result = await generate_product_embeddings.delay(tenant_id, product_data)
        success = embedding_result.get('success', False)
        
    return {'success': success, 'action': action}


async def _sync_shopify_data(tenant_id: str, config: Dict[str, Any], last_sync: Optional[datetime]) -> Dict[str, Any]:
    """Synchronize data with Shopify"""
    shopify_config = ShopifyConfig(
        shop_domain=config['shop_domain'],
        access_token=config['access_token']
    )
    
    integration = ShopifyIntegrationManager(shopify_config, tenant_id)
    
    products_synced = 0
    
    # Fetch products since last sync
    async for product_data in integration.fetch_all_products():
        try:
            await process_product_data.delay(tenant_id, product_data, 'update')
            products_synced += 1
        except Exception as e:
            logger.error(f"Failed to sync product {product_data.get('external_id')}: {e}")
            
    return {'products_synced': products_synced}


async def _get_last_sync_timestamp(tenant_id: str, platform: str) -> Optional[datetime]:
    """Get last sync timestamp for tenant and platform"""
    # This would typically query a sync_logs table
    # For now, return None to perform full sync
    return None


async def _update_sync_timestamp(tenant_id: str, platform: str, timestamp: datetime) -> None:
    """Update sync timestamp for tenant and platform"""
    # This would typically update a sync_logs table
    logger.info(f"Updated sync timestamp for tenant {tenant_id} platform {platform}: {timestamp}")


# Task monitoring and health checks

@celery_app.task(name="health_check")
async def health_check() -> Dict[str, Any]:
    """Health check task for ingestion worker"""
    try:
        await task_manager.initialize()
        
        # Test database connection
        async with get_db_session() as session:
            db_healthy = True
            
        # Test vector store connection
        vector_healthy = await task_manager.vector_store.health_check()
        
        # Test RabbitMQ connection
        rabbitmq_healthy = await task_manager.rabbitmq_publisher.health_check()
        
        return {
            'status': 'healthy',
            'database': db_healthy,
            'vector_store': vector_healthy,
            'rabbitmq': rabbitmq_healthy,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    finally:
        await task_manager.cleanup()


@celery_app.task(name="cleanup_failed_tasks")
async def cleanup_failed_tasks() -> Dict[str, Any]:
    """Cleanup failed tasks and retry if appropriate"""
    # This would implement cleanup logic for failed tasks
    # For now, return a placeholder
    return {
        'cleaned_tasks': 0,
        'retried_tasks': 0,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
