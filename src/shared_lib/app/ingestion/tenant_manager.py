"""
Tenant-Aware Data Ingestion Module

This module implements comprehensive tenant isolation and management
for data ingestion processes across all e-commerce platforms.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
import json
from dataclasses import dataclass
from enum import Enum

from ..schemas.models import Tenant, Product, EventLog
from ..schemas.events import TenantCreated, ProductIngested, DataIngestionCompleted
from ..db.database import get_db_session, BaseRepository
from ..utils.security import TenantSecurity, InputValidator
from ..utils.rabbitmq import RabbitMQPublisher
from ..storage.dual_storage import DualStorageManager, StorageOperation
from ..integrations.shopify import ShopifyIntegrationManager, ShopifyConfig

logger = logging.getLogger(__name__)


class TenantStatus(Enum):
    """Tenant status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    MAINTENANCE = "maintenance"


class IngestionStatus(Enum):
    """Data ingestion status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class TenantConfig:
    """Configuration for tenant data ingestion"""
    tenant_id: str
    platform: str
    platform_config: Dict[str, Any]
    ingestion_settings: Dict[str, Any]
    sync_schedule: Optional[str] = None
    webhook_endpoints: List[str] = None
    data_retention_days: int = 365
    max_products_per_sync: int = 10000


@dataclass
class IngestionJob:
    """Data ingestion job definition"""
    job_id: str
    tenant_id: str
    platform: str
    status: IngestionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    products_processed: int = 0
    products_created: int = 0
    products_updated: int = 0
    products_failed: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class TenantAwareIngestionManager:
    """
    Manages tenant-aware data ingestion with strict isolation
    """
    
    def __init__(self):
        self.tenant_security = TenantSecurity()
        self.rabbitmq_publisher = RabbitMQPublisher()
        self.active_jobs: Dict[str, IngestionJob] = {}
        self.tenant_configs: Dict[str, TenantConfig] = {}
        
    async def initialize(self) -> None:
        """Initialize the ingestion manager"""
        await self.rabbitmq_publisher.connect()
        await self._load_tenant_configurations()
        logger.info("Initialized tenant-aware ingestion manager")
        
    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.rabbitmq_publisher.disconnect()
        logger.info("Cleaned up tenant-aware ingestion manager")
        
    async def register_tenant(self, tenant_data: Dict[str, Any]) -> bool:
        """
        Register a new tenant for data ingestion
        
        Args:
            tenant_data: Tenant registration data
            
        Returns:
            True if registration successful
        """
        try:
            tenant_id = tenant_data['tenant_id']
            
            # Validate tenant data
            if not await self._validate_tenant_data(tenant_data):
                logger.error(f"Invalid tenant data for {tenant_id}")
                return False
                
            # Create tenant in database
            async with get_db_session() as session:
                repo = BaseRepository(session)
                
                # Check if tenant already exists
                existing_tenant = await repo.get_by_id(Tenant, tenant_id)
                if existing_tenant:
                    logger.warning(f"Tenant {tenant_id} already exists")
                    return False
                    
                # Create new tenant
                tenant = Tenant(
                    id=tenant_id,
                    name=tenant_data['name'],
                    email=tenant_data['email'],
                    platform=tenant_data['platform'],
                    status=TenantStatus.ACTIVE.value,
                    config=tenant_data.get('config', {}),
                    created_at=datetime.now(timezone.utc)
                )
                
                success = await repo.create(tenant, tenant_id)
                
                if success:
                    # Create tenant configuration
                    config = TenantConfig(
                        tenant_id=tenant_id,
                        platform=tenant_data['platform'],
                        platform_config=tenant_data.get('platform_config', {}),
                        ingestion_settings=tenant_data.get('ingestion_settings', {}),
                        sync_schedule=tenant_data.get('sync_schedule'),
                        webhook_endpoints=tenant_data.get('webhook_endpoints', []),
                        data_retention_days=tenant_data.get('data_retention_days', 365),
                        max_products_per_sync=tenant_data.get('max_products_per_sync', 10000)
                    )
                    
                    self.tenant_configs[tenant_id] = config
                    
                    # Publish tenant created event
                    tenant_event = TenantCreated(
                        tenant_id=tenant_id,
                        tenant_data=tenant_data,
                        timestamp=datetime.now(timezone.utc),
                        source='ingestion_manager'
                    )
                    
                    await self.rabbitmq_publisher.publish_event(
                        tenant_event, 
                        'tenant_created'
                    )
                    
                    logger.info(f"Successfully registered tenant {tenant_id}")
                    return True
                    
                return False
                
        except Exception as e:
            logger.error(f"Failed to register tenant {tenant_data.get('tenant_id', 'unknown')}: {e}")
            return False
            
    async def start_ingestion_job(self, tenant_id: str, platform: str, config: Dict[str, Any]) -> Optional[str]:
        """
        Start a data ingestion job for a tenant
        
        Args:
            tenant_id: Tenant identifier
            platform: E-commerce platform
            config: Platform-specific configuration
            
        Returns:
            Job ID if successful, None otherwise
        """
        try:
            # Validate tenant access
            if not await self.tenant_security.validate_tenant_access(tenant_id):
                logger.error(f"Invalid tenant access: {tenant_id}")
                return None
                
            # Check if tenant exists and is active
            tenant_status = await self._get_tenant_status(tenant_id)
            if tenant_status != TenantStatus.ACTIVE:
                logger.error(f"Tenant {tenant_id} is not active (status: {tenant_status})")
                return None
                
            # Check for existing active job
            if tenant_id in self.active_jobs:
                existing_job = self.active_jobs[tenant_id]
                if existing_job.status == IngestionStatus.IN_PROGRESS:
                    logger.warning(f"Active ingestion job already exists for tenant {tenant_id}")
                    return existing_job.job_id
                    
            # Create new job
            job_id = f"{tenant_id}_{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            job = IngestionJob(
                job_id=job_id,
                tenant_id=tenant_id,
                platform=platform,
                status=IngestionStatus.PENDING,
                started_at=datetime.now(timezone.utc),
                metadata={'config': config}
            )
            
            self.active_jobs[tenant_id] = job
            
            # Start ingestion process
            await self._execute_ingestion_job(job, config)
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start ingestion job for tenant {tenant_id}: {e}")
            return None
            
    async def get_ingestion_status(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get ingestion status for a tenant
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Status information or None
        """
        if tenant_id not in self.active_jobs:
            return None
            
        job = self.active_jobs[tenant_id]
        
        return {
            'job_id': job.job_id,
            'tenant_id': job.tenant_id,
            'platform': job.platform,
            'status': job.status.value,
            'started_at': job.started_at.isoformat(),
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'products_processed': job.products_processed,
            'products_created': job.products_created,
            'products_updated': job.products_updated,
            'products_failed': job.products_failed,
            'error_message': job.error_message,
            'progress_percentage': self._calculate_progress(job)
        }
        
    async def cancel_ingestion_job(self, tenant_id: str) -> bool:
        """
        Cancel an active ingestion job
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            True if cancellation successful
        """
        if tenant_id not in self.active_jobs:
            return False
            
        job = self.active_jobs[tenant_id]
        
        if job.status == IngestionStatus.IN_PROGRESS:
            job.status = IngestionStatus.FAILED
            job.error_message = "Job cancelled by user"
            job.completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Cancelled ingestion job for tenant {tenant_id}")
            return True
            
        return False
        
    async def sync_tenant_data(self, tenant_id: str, incremental: bool = True) -> Dict[str, Any]:
        """
        Synchronize tenant data with external platform
        
        Args:
            tenant_id: Tenant identifier
            incremental: Whether to perform incremental sync
            
        Returns:
            Sync results
        """
        try:
            # Validate tenant access
            if not await self.tenant_security.validate_tenant_access(tenant_id):
                raise ValueError(f"Invalid tenant access: {tenant_id}")
                
            # Get tenant configuration
            if tenant_id not in self.tenant_configs:
                raise ValueError(f"No configuration found for tenant {tenant_id}")
                
            config = self.tenant_configs[tenant_id]
            
            # Initialize storage manager
            storage_manager = DualStorageManager(tenant_id)
            await storage_manager.initialize()
            
            try:
                # Route to platform-specific sync
                if config.platform == 'shopify':
                    result = await self._sync_shopify_data(tenant_id, config, incremental)
                else:
                    raise ValueError(f"Unsupported platform: {config.platform}")
                    
                # Update sync timestamp
                await self._update_sync_timestamp(tenant_id, config.platform)
                
                return result
                
            finally:
                await storage_manager.cleanup()
                
        except Exception as e:
            logger.error(f"Failed to sync data for tenant {tenant_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
    async def handle_webhook(self, tenant_id: str, payload: bytes, signature: str, topic: str) -> bool:
        """
        Handle incoming webhook for tenant
        
        Args:
            tenant_id: Tenant identifier
            payload: Webhook payload
            signature: Webhook signature
            topic: Webhook topic
            
        Returns:
            True if webhook handled successfully
        """
        try:
            # Validate tenant access
            if not await self.tenant_security.validate_tenant_access(tenant_id):
                logger.error(f"Invalid tenant access for webhook: {tenant_id}")
                return False
                
            # Get tenant configuration
            if tenant_id not in self.tenant_configs:
                logger.error(f"No configuration found for tenant {tenant_id}")
                return False
                
            config = self.tenant_configs[tenant_id]
            
            # Route to platform-specific webhook handler
            if config.platform == 'shopify':
                return await self._handle_shopify_webhook(tenant_id, config, payload, signature, topic)
            else:
                logger.warning(f"Unsupported platform for webhook: {config.platform}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to handle webhook for tenant {tenant_id}: {e}")
            return False
            
    async def get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get comprehensive metrics for a tenant
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Tenant metrics
        """
        try:
            # Validate tenant access
            if not await self.tenant_security.validate_tenant_access(tenant_id):
                raise ValueError(f"Invalid tenant access: {tenant_id}")
                
            # Get product count
            async with get_db_session() as session:
                repo = BaseRepository(session)
                products = await repo.get_all_by_tenant(Product, tenant_id)
                
            # Get ingestion job history
            job_history = await self._get_job_history(tenant_id)
            
            # Get storage health
            storage_manager = DualStorageManager(tenant_id)
            await storage_manager.initialize()
            
            try:
                storage_health = await storage_manager.get_storage_health()
            finally:
                await storage_manager.cleanup()
                
            return {
                'tenant_id': tenant_id,
                'total_products': len(products),
                'active_products': len([p for p in products if p.status == 'active']),
                'last_sync': await self._get_last_sync_timestamp(tenant_id),
                'job_history': job_history,
                'storage_health': storage_health,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics for tenant {tenant_id}: {e}")
            return {'error': str(e)}
            
    async def _load_tenant_configurations(self) -> None:
        """Load tenant configurations from database"""
        try:
            async with get_db_session() as session:
                repo = BaseRepository(session)
                tenants = await repo.get_all(Tenant)
                
            for tenant in tenants:
                config = TenantConfig(
                    tenant_id=tenant.id,
                    platform=tenant.platform,
                    platform_config=tenant.config.get('platform_config', {}),
                    ingestion_settings=tenant.config.get('ingestion_settings', {}),
                    sync_schedule=tenant.config.get('sync_schedule'),
                    webhook_endpoints=tenant.config.get('webhook_endpoints', []),
                    data_retention_days=tenant.config.get('data_retention_days', 365),
                    max_products_per_sync=tenant.config.get('max_products_per_sync', 10000)
                )
                
                self.tenant_configs[tenant.id] = config
                
            logger.info(f"Loaded configurations for {len(tenants)} tenants")
            
        except Exception as e:
            logger.error(f"Failed to load tenant configurations: {e}")
            
    async def _validate_tenant_data(self, tenant_data: Dict[str, Any]) -> bool:
        """Validate tenant registration data"""
        required_fields = ['tenant_id', 'name', 'email', 'platform']
        
        for field in required_fields:
            if field not in tenant_data or not tenant_data[field]:
                logger.error(f"Missing required field: {field}")
                return False
                
        # Validate email format
        email = tenant_data['email']
        if '@' not in email or '.' not in email.split('@')[1]:
            logger.error("Invalid email format")
            return False
            
        # Validate platform
        supported_platforms = ['shopify', 'woocommerce', 'magento']
        if tenant_data['platform'] not in supported_platforms:
            logger.error(f"Unsupported platform: {tenant_data['platform']}")
            return False
            
        return True
        
    async def _get_tenant_status(self, tenant_id: str) -> TenantStatus:
        """Get tenant status"""
        try:
            async with get_db_session() as session:
                repo = BaseRepository(session)
                tenant = await repo.get_by_id(Tenant, tenant_id)
                
            if tenant:
                return TenantStatus(tenant.status)
            else:
                return TenantStatus.INACTIVE
                
        except Exception as e:
            logger.error(f"Failed to get tenant status: {e}")
            return TenantStatus.INACTIVE
            
    async def _execute_ingestion_job(self, job: IngestionJob, config: Dict[str, Any]) -> None:
        """Execute ingestion job"""
        try:
            job.status = IngestionStatus.IN_PROGRESS
            
            # Initialize storage manager
            storage_manager = DualStorageManager(job.tenant_id)
            await storage_manager.initialize()
            
            try:
                # Route to platform-specific ingestion
                if job.platform == 'shopify':
                    result = await self._ingest_shopify_products(job, config, storage_manager)
                else:
                    raise ValueError(f"Unsupported platform: {job.platform}")
                    
                # Update job status
                job.products_processed = result['products_processed']
                job.products_created = result['products_created']
                job.products_updated = result['products_updated']
                job.products_failed = result['products_failed']
                
                if job.products_failed == 0:
                    job.status = IngestionStatus.COMPLETED
                elif job.products_processed > 0:
                    job.status = IngestionStatus.PARTIAL
                else:
                    job.status = IngestionStatus.FAILED
                    job.error_message = "No products processed"
                    
                job.completed_at = datetime.now(timezone.utc)
                
                # Publish completion event
                completion_event = DataIngestionCompleted(
                    tenant_id=job.tenant_id,
                    platform=job.platform,
                    task_id=job.job_id,
                    products_processed=job.products_processed,
                    products_created=job.products_created,
                    products_updated=job.products_updated,
                    products_failed=job.products_failed,
                    vectors_created=result.get('vectors_created', 0),
                    timestamp=datetime.now(timezone.utc),
                    status=job.status.value
                )
                
                await self.rabbitmq_publisher.publish_event(
                    completion_event, 
                    'data_ingestion_completed'
                )
                
            finally:
                await storage_manager.cleanup()
                
        except Exception as e:
            logger.error(f"Failed to execute ingestion job {job.job_id}: {e}")
            job.status = IngestionStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            
    async def _ingest_shopify_products(self, job: IngestionJob, config: Dict[str, Any], storage_manager: DualStorageManager) -> Dict[str, Any]:
        """Ingest products from Shopify"""
        shopify_config = ShopifyConfig(
            shop_domain=config['shop_domain'],
            access_token=config['access_token'],
            webhook_secret=config.get('webhook_secret')
        )
        
        integration = ShopifyIntegrationManager(shopify_config, job.tenant_id)
        
        products_processed = 0
        products_created = 0
        products_updated = 0
        products_failed = 0
        vectors_created = 0
        
        # Get tenant configuration limits
        tenant_config = self.tenant_configs.get(job.tenant_id)
        max_products = tenant_config.max_products_per_sync if tenant_config else 10000
        
        async for product_data in integration.fetch_all_products():
            if products_processed >= max_products:
                logger.warning(f"Reached max products limit ({max_products}) for tenant {job.tenant_id}")
                break
                
            try:
                # Store product with tenant isolation
                result = await storage_manager.store_product(product_data, StorageOperation.UPSERT)
                
                if result.success:
                    products_processed += 1
                    
                    if result.database_result and result.database_result.get('operation') == 'create':
                        products_created += 1
                    else:
                        products_updated += 1
                        
                    if result.vector_result and result.vector_result.get('success'):
                        vectors_created += 1
                else:
                    products_failed += 1
                    
            except Exception as e:
                logger.error(f"Failed to process product {product_data.get('external_id', 'unknown')}: {e}")
                products_failed += 1
                
        return {
            'products_processed': products_processed,
            'products_created': products_created,
            'products_updated': products_updated,
            'products_failed': products_failed,
            'vectors_created': vectors_created
        }
        
    async def _sync_shopify_data(self, tenant_id: str, config: TenantConfig, incremental: bool) -> Dict[str, Any]:
        """Synchronize data with Shopify"""
        shopify_config = ShopifyConfig(
            shop_domain=config.platform_config['shop_domain'],
            access_token=config.platform_config['access_token']
        )
        
        integration = ShopifyIntegrationManager(shopify_config, tenant_id)
        
        products_synced = 0
        
        # Get last sync timestamp for incremental sync
        last_sync = None
        if incremental:
            last_sync = await self._get_last_sync_timestamp(tenant_id)
            
        # Fetch products since last sync
        async for product_data in integration.fetch_all_products():
            try:
                storage_manager = DualStorageManager(tenant_id)
                await storage_manager.initialize()
                
                try:
                    await storage_manager.store_product(product_data, StorageOperation.UPDATE)
                    products_synced += 1
                finally:
                    await storage_manager.cleanup()
                    
            except Exception as e:
                logger.error(f"Failed to sync product {product_data.get('external_id', 'unknown')}: {e}")
                
        return {
            'products_synced': products_synced,
            'incremental': incremental,
            'last_sync': last_sync.isoformat() if last_sync else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    async def _handle_shopify_webhook(self, tenant_id: str, config: TenantConfig, payload: bytes, signature: str, topic: str) -> bool:
        """Handle Shopify webhook"""
        try:
            shopify_config = ShopifyConfig(
                shop_domain=config.platform_config['shop_domain'],
                access_token=config.platform_config['access_token'],
                webhook_secret=config.platform_config.get('webhook_secret')
            )
            
            integration = ShopifyIntegrationManager(shopify_config, tenant_id)
            
            # Handle webhook
            event = await integration.handle_webhook(payload, signature, topic)
            
            if event:
                # Process the event
                storage_manager = DualStorageManager(tenant_id)
                await storage_manager.initialize()
                
                try:
                    if event.action == 'deleted':
                        await storage_manager.delete_product(event.product_id)
                    else:
                        await storage_manager.store_product(event.product_data, StorageOperation.UPSERT)
                        
                    # Publish event
                    await self.rabbitmq_publisher.publish_event(event, 'product_ingested')
                    
                    return True
                    
                finally:
                    await storage_manager.cleanup()
                    
            return False
            
        except Exception as e:
            logger.error(f"Failed to handle Shopify webhook: {e}")
            return False
            
    def _calculate_progress(self, job: IngestionJob) -> float:
        """Calculate job progress percentage"""
        if job.status == IngestionStatus.COMPLETED:
            return 100.0
        elif job.status == IngestionStatus.FAILED:
            return 0.0
        elif job.status == IngestionStatus.IN_PROGRESS:
            # Estimate progress based on time elapsed
            elapsed = datetime.now(timezone.utc) - job.started_at
            estimated_duration = 300  # 5 minutes estimate
            progress = min(90.0, (elapsed.total_seconds() / estimated_duration) * 100)
            return progress
        else:
            return 0.0
            
    async def _get_job_history(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get job history for tenant"""
        # This would typically query a jobs table
        # For now, return current active job if exists
        if tenant_id in self.active_jobs:
            job = self.active_jobs[tenant_id]
            return [{
                'job_id': job.job_id,
                'status': job.status.value,
                'started_at': job.started_at.isoformat(),
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'products_processed': job.products_processed
            }]
        return []
        
    async def _get_last_sync_timestamp(self, tenant_id: str) -> Optional[datetime]:
        """Get last sync timestamp for tenant"""
        # This would typically query a sync_logs table
        return None
        
    async def _update_sync_timestamp(self, tenant_id: str, platform: str) -> None:
        """Update sync timestamp for tenant"""
        # This would typically update a sync_logs table
        logger.info(f"Updated sync timestamp for tenant {tenant_id} platform {platform}")


class TenantDataValidator:
    """
    Validates tenant data integrity and consistency
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.tenant_security = TenantSecurity()
        
    async def validate_tenant_data_integrity(self) -> Dict[str, Any]:
        """Validate tenant data integrity"""
        try:
            # Initialize storage manager
            storage_manager = DualStorageManager(self.tenant_id)
            await storage_manager.initialize()
            
            try:
                # Get storage health
                health = await storage_manager.get_storage_health()
                
                # Validate data consistency
                from ..storage.dual_storage import StorageSynchronizer
                synchronizer = StorageSynchronizer(self.tenant_id)
                consistency = await synchronizer.validate_data_consistency()
                
                return {
                    'tenant_id': self.tenant_id,
                    'storage_health': health,
                    'data_consistency': consistency,
                    'validation_timestamp': datetime.now(timezone.utc).isoformat()
                }
                
            finally:
                await storage_manager.cleanup()
                
        except Exception as e:
            logger.error(f"Failed to validate tenant data integrity: {e}")
            return {
                'tenant_id': self.tenant_id,
                'error': str(e),
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }
