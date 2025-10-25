"""
Data Synchronization and Update Mechanisms Module

This module implements comprehensive data synchronization, conflict resolution,
and update mechanisms for maintaining data consistency across platforms.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timezone, timedelta
import json
import hashlib
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, func

from ..schemas.models import Product, Tenant, EventLog
from ..schemas.events import ProductIngested, DataIngestionCompleted, DataSyncCompleted
from ..db.database import get_db_session, BaseRepository
from ..utils.security import TenantSecurity
from ..utils.rabbitmq import RabbitMQPublisher
from ..storage.dual_storage import DualStorageManager, StorageOperation, StorageSynchronizer
from ..integrations.shopify import ShopifyIntegrationManager, ShopifyConfig

logger = logging.getLogger(__name__)


class SyncStrategy(Enum):
    """Synchronization strategy enumeration"""
    FULL_SYNC = "full_sync"
    INCREMENTAL_SYNC = "incremental_sync"
    DIFFERENTIAL_SYNC = "differential_sync"
    REAL_TIME_SYNC = "real_time_sync"


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategy enumeration"""
    PLATFORM_WINS = "platform_wins"
    DATABASE_WINS = "database_wins"
    NEWER_WINS = "newer_wins"
    MANUAL_RESOLUTION = "manual_resolution"
    MERGE_STRATEGY = "merge_strategy"


class SyncStatus(Enum):
    """Synchronization status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CONFLICT = "conflict"


@dataclass
class SyncJob:
    """Synchronization job definition"""
    job_id: str
    tenant_id: str
    platform: str
    strategy: SyncStrategy
    status: SyncStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    products_synced: int = 0
    products_created: int = 0
    products_updated: int = 0
    products_deleted: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class DataConflict:
    """Data conflict definition"""
    conflict_id: str
    tenant_id: str
    product_id: str
    conflict_type: str
    platform_data: Dict[str, Any]
    database_data: Dict[str, Any]
    detected_at: datetime
    resolution_strategy: ConflictResolutionStrategy
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_data: Optional[Dict[str, Any]] = None


class DataSynchronizationManager:
    """
    Manages data synchronization across platforms with conflict resolution
    """
    
    def __init__(self):
        self.tenant_security = TenantSecurity()
        self.rabbitmq_publisher = RabbitMQPublisher()
        self.active_sync_jobs: Dict[str, SyncJob] = {}
        self.data_conflicts: Dict[str, DataConflict] = {}
        self.sync_schedules: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> None:
        """Initialize synchronization manager"""
        await self.rabbitmq_publisher.connect()
        await self._load_sync_schedules()
        logger.info("Initialized data synchronization manager")
        
    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.rabbitmq_publisher.disconnect()
        logger.info("Cleaned up data synchronization manager")
        
    async def start_sync_job(
        self, 
        tenant_id: str, 
        platform: str, 
        strategy: SyncStrategy = SyncStrategy.INCREMENTAL_SYNC,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Start a synchronization job
        
        Args:
            tenant_id: Tenant identifier
            platform: E-commerce platform
            strategy: Synchronization strategy
            config: Platform configuration
            
        Returns:
            Job ID if successful, None otherwise
        """
        try:
            # Validate tenant access
            if not await self.tenant_security.validate_tenant_access(tenant_id):
                logger.error(f"Invalid tenant access: {tenant_id}")
                return None
                
            # Check for existing active sync job
            if tenant_id in self.active_sync_jobs:
                existing_job = self.active_sync_jobs[tenant_id]
                if existing_job.status == SyncStatus.IN_PROGRESS:
                    logger.warning(f"Active sync job already exists for tenant {tenant_id}")
                    return existing_job.job_id
                    
            # Create new sync job
            job_id = f"sync_{tenant_id}_{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            job = SyncJob(
                job_id=job_id,
                tenant_id=tenant_id,
                platform=platform,
                strategy=strategy,
                status=SyncStatus.PENDING,
                started_at=datetime.now(timezone.utc),
                metadata={'config': config or {}}
            )
            
            self.active_sync_jobs[tenant_id] = job
            
            # Start synchronization process
            await self._execute_sync_job(job)
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start sync job for tenant {tenant_id}: {e}")
            return None
            
    async def get_sync_status(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get synchronization status for tenant"""
        if tenant_id not in self.active_sync_jobs:
            return None
            
        job = self.active_sync_jobs[tenant_id]
        
        return {
            'job_id': job.job_id,
            'tenant_id': job.tenant_id,
            'platform': job.platform,
            'strategy': job.strategy.value,
            'status': job.status.value,
            'started_at': job.started_at.isoformat(),
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'products_synced': job.products_synced,
            'products_created': job.products_created,
            'products_updated': job.products_updated,
            'products_deleted': job.products_deleted,
            'conflicts_detected': job.conflicts_detected,
            'conflicts_resolved': job.conflicts_resolved,
            'error_message': job.error_message,
            'progress_percentage': self._calculate_sync_progress(job)
        }
        
    async def resolve_data_conflict(
        self, 
        conflict_id: str, 
        resolution_strategy: ConflictResolutionStrategy,
        resolution_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Resolve a data conflict
        
        Args:
            conflict_id: Conflict identifier
            resolution_strategy: Resolution strategy
            resolution_data: Custom resolution data
            
        Returns:
            True if resolution successful
        """
        try:
            if conflict_id not in self.data_conflicts:
                logger.error(f"Conflict {conflict_id} not found")
                return False
                
            conflict = self.data_conflicts[conflict_id]
            
            # Apply resolution strategy
            resolved_data = await self._apply_resolution_strategy(
                conflict, 
                resolution_strategy, 
                resolution_data
            )
            
            if resolved_data:
                # Update the resolved data
                storage_manager = DualStorageManager(conflict.tenant_id)
                await storage_manager.initialize()
                
                try:
                    await storage_manager.store_product(resolved_data, StorageOperation.UPDATE)
                    
                    # Mark conflict as resolved
                    conflict.resolved = True
                    conflict.resolved_at = datetime.now(timezone.utc)
                    conflict.resolution_strategy = resolution_strategy
                    conflict.resolution_data = resolved_data
                    
                    logger.info(f"Resolved conflict {conflict_id} using {resolution_strategy.value}")
                    return True
                    
                finally:
                    await storage_manager.cleanup()
                    
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve conflict {conflict_id}: {e}")
            return False
            
    async def schedule_automatic_sync(
        self, 
        tenant_id: str, 
        platform: str, 
        schedule_cron: str,
        strategy: SyncStrategy = SyncStrategy.INCREMENTAL_SYNC
    ) -> bool:
        """
        Schedule automatic synchronization
        
        Args:
            tenant_id: Tenant identifier
            platform: E-commerce platform
            schedule_cron: Cron expression for scheduling
            strategy: Synchronization strategy
            
        Returns:
            True if scheduling successful
        """
        try:
            # Validate tenant access
            if not await self.tenant_security.validate_tenant_access(tenant_id):
                logger.error(f"Invalid tenant access: {tenant_id}")
                return False
                
            # Store schedule
            schedule_key = f"{tenant_id}_{platform}"
            self.sync_schedules[schedule_key] = {
                'tenant_id': tenant_id,
                'platform': platform,
                'schedule_cron': schedule_cron,
                'strategy': strategy,
                'enabled': True,
                'created_at': datetime.now(timezone.utc),
                'last_run': None,
                'next_run': self._calculate_next_run(schedule_cron)
            }
            
            logger.info(f"Scheduled automatic sync for tenant {tenant_id} platform {platform}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule automatic sync: {e}")
            return False
            
    async def detect_data_drift(self, tenant_id: str) -> Dict[str, Any]:
        """
        Detect data drift between platform and database
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Data drift analysis results
        """
        try:
            # Validate tenant access
            if not await self.tenant_security.validate_tenant_access(tenant_id):
                raise ValueError(f"Invalid tenant access: {tenant_id}")
                
            # Get tenant configuration
            async with get_db_session() as session:
                repo = BaseRepository(session)
                tenant = await repo.get_by_id(Tenant, tenant_id)
                
            if not tenant:
                raise ValueError(f"Tenant {tenant_id} not found")
                
            # Initialize storage manager
            storage_manager = DualStorageManager(tenant_id)
            await storage_manager.initialize()
            
            try:
                # Get all products from database
                async with get_db_session() as session:
                    repo = BaseRepository(session)
                    db_products = await repo.get_all_by_tenant(Product, tenant_id)
                    
                db_product_ids = {p.external_id for p in db_products}
                
                # Get products from platform
                platform_products = await self._fetch_platform_products(tenant_id, tenant.platform)
                platform_product_ids = {p['external_id'] for p in platform_products}
                
                # Analyze differences
                missing_in_platform = db_product_ids - platform_product_ids
                missing_in_database = platform_product_ids - db_product_ids
                common_products = db_product_ids & platform_product_ids
                
                # Check for data inconsistencies in common products
                inconsistencies = []
                for product_id in list(common_products)[:100]:  # Sample check
                    db_product = next(p for p in db_products if p.external_id == product_id)
                    platform_product = next(p for p in platform_products if p['external_id'] == product_id)
                    
                    inconsistency = await self._detect_product_inconsistency(
                        db_product, 
                        platform_product
                    )
                    
                    if inconsistency:
                        inconsistencies.append(inconsistency)
                        
                return {
                    'tenant_id': tenant_id,
                    'total_db_products': len(db_products),
                    'total_platform_products': len(platform_products),
                    'missing_in_platform': list(missing_in_platform),
                    'missing_in_database': list(missing_in_database),
                    'common_products': len(common_products),
                    'inconsistencies_detected': len(inconsistencies),
                    'inconsistencies': inconsistencies[:10],  # Limit results
                    'drift_percentage': (
                        len(missing_in_platform) + len(missing_in_database) + len(inconsistencies)
                    ) / max(len(db_products), len(platform_products)) * 100,
                    'analysis_timestamp': datetime.now(timezone.utc).isoformat()
                }
                
            finally:
                await storage_manager.cleanup()
                
        except Exception as e:
            logger.error(f"Failed to detect data drift for tenant {tenant_id}: {e}")
            return {'error': str(e)}
            
    async def perform_data_migration(
        self, 
        tenant_id: str, 
        source_platform: str, 
        target_platform: str,
        migration_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform data migration between platforms
        
        Args:
            tenant_id: Tenant identifier
            source_platform: Source platform
            target_platform: Target platform
            migration_config: Migration configuration
            
        Returns:
            Migration results
        """
        try:
            # Validate tenant access
            if not await self.tenant_security.validate_tenant_access(tenant_id):
                raise ValueError(f"Invalid tenant access: {tenant_id}")
                
            migration_id = f"migration_{tenant_id}_{source_platform}_to_{target_platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting data migration {migration_id}")
            
            # Get products from source platform
            source_products = await self._fetch_platform_products(tenant_id, source_platform)
            
            # Transform products for target platform
            transformed_products = await self._transform_products_for_platform(
                source_products, 
                target_platform, 
                migration_config
            )
            
            # Store in target platform (this would typically involve API calls)
            migration_results = await self._migrate_products_to_platform(
                tenant_id, 
                target_platform, 
                transformed_products
            )
            
            # Update database with new platform data
            storage_manager = DualStorageManager(tenant_id)
            await storage_manager.initialize()
            
            try:
                for product in transformed_products:
                    await storage_manager.store_product(product, StorageOperation.UPSERT)
                    
            finally:
                await storage_manager.cleanup()
                
            return {
                'migration_id': migration_id,
                'tenant_id': tenant_id,
                'source_platform': source_platform,
                'target_platform': target_platform,
                'products_migrated': len(transformed_products),
                'migration_results': migration_results,
                'completed_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to perform data migration: {e}")
            return {'error': str(e)}
            
    async def _execute_sync_job(self, job: SyncJob) -> None:
        """Execute synchronization job"""
        try:
            job.status = SyncStatus.IN_PROGRESS
            
            # Initialize storage manager
            storage_manager = DualStorageManager(job.tenant_id)
            await storage_manager.initialize()
            
            try:
                # Execute sync based on strategy
                if job.strategy == SyncStrategy.FULL_SYNC:
                    result = await self._perform_full_sync(job, storage_manager)
                elif job.strategy == SyncStrategy.INCREMENTAL_SYNC:
                    result = await self._perform_incremental_sync(job, storage_manager)
                elif job.strategy == SyncStrategy.DIFFERENTIAL_SYNC:
                    result = await self._perform_differential_sync(job, storage_manager)
                else:
                    raise ValueError(f"Unsupported sync strategy: {job.strategy}")
                    
                # Update job status
                job.products_synced = result['products_synced']
                job.products_created = result['products_created']
                job.products_updated = result['products_updated']
                job.products_deleted = result['products_deleted']
                job.conflicts_detected = result['conflicts_detected']
                job.conflicts_resolved = result['conflicts_resolved']
                
                if job.conflicts_detected > 0 and job.conflicts_resolved < job.conflicts_detected:
                    job.status = SyncStatus.CONFLICT
                elif job.products_synced > 0:
                    job.status = SyncStatus.COMPLETED
                else:
                    job.status = SyncStatus.FAILED
                    job.error_message = "No products synced"
                    
                job.completed_at = datetime.now(timezone.utc)
                
                # Publish completion event
                sync_event = DataSyncCompleted(
                    tenant_id=job.tenant_id,
                    platform=job.platform,
                    sync_job_id=job.job_id,
                    strategy=job.strategy.value,
                    products_synced=job.products_synced,
                    conflicts_detected=job.conflicts_detected,
                    conflicts_resolved=job.conflicts_resolved,
                    timestamp=datetime.now(timezone.utc),
                    status=job.status.value
                )
                
                await self.rabbitmq_publisher.publish_event(
                    sync_event, 
                    'data_sync_completed'
                )
                
            finally:
                await storage_manager.cleanup()
                
        except Exception as e:
            logger.error(f"Failed to execute sync job {job.job_id}: {e}")
            job.status = SyncStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            
    async def _perform_full_sync(self, job: SyncJob, storage_manager: DualStorageManager) -> Dict[str, Any]:
        """Perform full synchronization"""
        # Get all products from platform
        platform_products = await self._fetch_platform_products(job.tenant_id, job.platform)
        
        products_synced = 0
        products_created = 0
        products_updated = 0
        conflicts_detected = 0
        conflicts_resolved = 0
        
        for product_data in platform_products:
            try:
                # Check for conflicts
                conflict = await self._detect_conflict(job.tenant_id, product_data)
                
                if conflict:
                    conflicts_detected += 1
                    # Auto-resolve using newer_wins strategy
                    resolved_data = await self._apply_resolution_strategy(
                        conflict, 
                        ConflictResolutionStrategy.NEWER_WINS
                    )
                    
                    if resolved_data:
                        await storage_manager.store_product(resolved_data, StorageOperation.UPSERT)
                        conflicts_resolved += 1
                else:
                    # No conflict, proceed with sync
                    result = await storage_manager.store_product(product_data, StorageOperation.UPSERT)
                    
                    if result.success:
                        if result.database_result and result.database_result.get('operation') == 'create':
                            products_created += 1
                        else:
                            products_updated += 1
                            
                products_synced += 1
                
            except Exception as e:
                logger.error(f"Failed to sync product {product_data.get('external_id', 'unknown')}: {e}")
                
        return {
            'products_synced': products_synced,
            'products_created': products_created,
            'products_updated': products_updated,
            'products_deleted': 0,
            'conflicts_detected': conflicts_detected,
            'conflicts_resolved': conflicts_resolved
        }
        
    async def _perform_incremental_sync(self, job: SyncJob, storage_manager: DualStorageManager) -> Dict[str, Any]:
        """Perform incremental synchronization"""
        # Get last sync timestamp
        last_sync = await self._get_last_sync_timestamp(job.tenant_id, job.platform)
        
        # Get products updated since last sync
        platform_products = await self._fetch_platform_products_since(
            job.tenant_id, 
            job.platform, 
            last_sync
        )
        
        # Process products (similar to full sync but with timestamp filtering)
        return await self._process_sync_products(job, platform_products, storage_manager)
        
    async def _perform_differential_sync(self, job: SyncJob, storage_manager: DualStorageManager) -> Dict[str, Any]:
        """Perform differential synchronization"""
        # Get data drift analysis
        drift_analysis = await self.detect_data_drift(job.tenant_id)
        
        # Sync only products with differences
        products_to_sync = []
        
        # Add missing products
        for product_id in drift_analysis.get('missing_in_database', []):
            product_data = await self._fetch_product_by_id(job.tenant_id, job.platform, product_id)
            if product_data:
                products_to_sync.append(product_data)
                
        # Add inconsistent products
        for inconsistency in drift_analysis.get('inconsistencies', []):
            product_data = await self._fetch_product_by_id(
                job.tenant_id, 
                job.platform, 
                inconsistency['product_id']
            )
            if product_data:
                products_to_sync.append(product_data)
                
        return await self._process_sync_products(job, products_to_sync, storage_manager)
        
    async def _process_sync_products(
        self, 
        job: SyncJob, 
        products: List[Dict[str, Any]], 
        storage_manager: DualStorageManager
    ) -> Dict[str, Any]:
        """Process products for synchronization"""
        products_synced = 0
        products_created = 0
        products_updated = 0
        conflicts_detected = 0
        conflicts_resolved = 0
        
        for product_data in products:
            try:
                result = await storage_manager.store_product(product_data, StorageOperation.UPSERT)
                
                if result.success:
                    products_synced += 1
                    
                    if result.database_result and result.database_result.get('operation') == 'create':
                        products_created += 1
                    else:
                        products_updated += 1
                        
            except Exception as e:
                logger.error(f"Failed to sync product {product_data.get('external_id', 'unknown')}: {e}")
                
        return {
            'products_synced': products_synced,
            'products_created': products_created,
            'products_updated': products_updated,
            'products_deleted': 0,
            'conflicts_detected': conflicts_detected,
            'conflicts_resolved': conflicts_resolved
        }
        
    async def _detect_conflict(self, tenant_id: str, platform_data: Dict[str, Any]) -> Optional[DataConflict]:
        """Detect data conflict between platform and database"""
        try:
            # Get product from database
            storage_manager = DualStorageManager(tenant_id)
            await storage_manager.initialize()
            
            try:
                db_product = await storage_manager.retrieve_product(platform_data['external_id'])
                
                if db_product:
                    # Compare data
                    platform_hash = self._calculate_data_hash(platform_data)
                    db_hash = self._calculate_data_hash(db_product)
                    
                    if platform_hash != db_hash:
                        # Conflict detected
                        conflict_id = f"conflict_{tenant_id}_{platform_data['external_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        conflict = DataConflict(
                            conflict_id=conflict_id,
                            tenant_id=tenant_id,
                            product_id=platform_data['external_id'],
                            conflict_type='data_mismatch',
                            platform_data=platform_data,
                            database_data=db_product,
                            detected_at=datetime.now(timezone.utc),
                            resolution_strategy=ConflictResolutionStrategy.NEWER_WINS
                        )
                        
                        self.data_conflicts[conflict_id] = conflict
                        return conflict
                        
            finally:
                await storage_manager.cleanup()
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect conflict: {e}")
            return None
            
    async def _apply_resolution_strategy(
        self, 
        conflict: DataConflict, 
        strategy: ConflictResolutionStrategy,
        resolution_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Apply conflict resolution strategy"""
        try:
            if strategy == ConflictResolutionStrategy.PLATFORM_WINS:
                return conflict.platform_data
                
            elif strategy == ConflictResolutionStrategy.DATABASE_WINS:
                return conflict.database_data
                
            elif strategy == ConflictResolutionStrategy.NEWER_WINS:
                platform_time = self._extract_timestamp(conflict.platform_data)
                db_time = self._extract_timestamp(conflict.database_data)
                
                if platform_time and db_time:
                    return conflict.platform_data if platform_time > db_time else conflict.database_data
                else:
                    return conflict.platform_data  # Default to platform
                    
            elif strategy == ConflictResolutionStrategy.MERGE_STRATEGY:
                return await self._merge_product_data(
                    conflict.platform_data, 
                    conflict.database_data
                )
                
            elif strategy == ConflictResolutionStrategy.MANUAL_RESOLUTION:
                return resolution_data
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to apply resolution strategy: {e}")
            return None
            
    async def _merge_product_data(
        self, 
        platform_data: Dict[str, Any], 
        database_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge product data from platform and database"""
        merged_data = database_data.copy()
        
        # Merge fields, preferring platform data for most fields
        merge_fields = ['title', 'description', 'price', 'compare_at_price', 'inventory_quantity']
        
        for field in merge_fields:
            if field in platform_data and platform_data[field] is not None:
                merged_data[field] = platform_data[field]
                
        # Merge metadata
        if 'metadata' in platform_data:
            merged_data['metadata'] = {
                **merged_data.get('metadata', {}),
                **platform_data['metadata'],
                'merged_at': datetime.now(timezone.utc).isoformat()
            }
            
        return merged_data
        
    def _calculate_data_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash for data comparison"""
        # Create a normalized version for hashing
        normalized_data = {
            'title': data.get('title', ''),
            'description': data.get('description', ''),
            'price': data.get('price', ''),
            'compare_at_price': data.get('compare_at_price', ''),
            'inventory_quantity': data.get('inventory_quantity', 0),
            'status': data.get('status', ''),
            'updated_at': data.get('updated_at', '')
        }
        
        data_string = json.dumps(normalized_data, sort_keys=True)
        return hashlib.md5(data_string.encode('utf-8')).hexdigest()
        
    def _extract_timestamp(self, data: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from product data"""
        timestamp_fields = ['updated_at', 'created_at', 'published_at']
        
        for field in timestamp_fields:
            if field in data and data[field]:
                try:
                    if isinstance(data[field], str):
                        return datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                    elif isinstance(data[field], datetime):
                        return data[field]
                except (ValueError, TypeError):
                    continue
                    
        return None
        
    async def _fetch_platform_products(self, tenant_id: str, platform: str) -> List[Dict[str, Any]]:
        """Fetch products from platform"""
        try:
            async with get_db_session() as session:
                repo = BaseRepository(session)
                tenant = await repo.get_by_id(Tenant, tenant_id)
                
            if not tenant:
                raise ValueError(f"Tenant {tenant_id} not found")
                
            if platform == 'shopify':
                shopify_config = ShopifyConfig(
                    shop_domain=tenant.config['platform_config']['shop_domain'],
                    access_token=tenant.config['platform_config']['access_token']
                )
                
                integration = ShopifyIntegrationManager(shopify_config, tenant_id)
                
                products = []
                async for product in integration.fetch_all_products():
                    products.append(product)
                    
                return products
            else:
                raise ValueError(f"Unsupported platform: {platform}")
                
        except Exception as e:
            logger.error(f"Failed to fetch platform products: {e}")
            return []
            
    async def _fetch_platform_products_since(
        self, 
        tenant_id: str, 
        platform: str, 
        since: Optional[datetime]
    ) -> List[Dict[str, Any]]:
        """Fetch products updated since timestamp"""
        # This would typically use platform-specific APIs to filter by timestamp
        # For now, return all products
        return await self._fetch_platform_products(tenant_id, platform)
        
    async def _fetch_product_by_id(self, tenant_id: str, platform: str, product_id: str) -> Optional[Dict[str, Any]]:
        """Fetch specific product by ID from platform"""
        try:
            async with get_db_session() as session:
                repo = BaseRepository(session)
                tenant = await repo.get_by_id(Tenant, tenant_id)
                
            if not tenant:
                return None
                
            if platform == 'shopify':
                shopify_config = ShopifyConfig(
                    shop_domain=tenant.config['platform_config']['shop_domain'],
                    access_token=tenant.config['platform_config']['access_token']
                )
                
                integration = ShopifyIntegrationManager(shopify_config, tenant_id)
                return await integration.fetch_product_by_id(int(product_id))
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch product {product_id}: {e}")
            return None
            
    async def _detect_product_inconsistency(
        self, 
        db_product: Product, 
        platform_product: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect inconsistency between database and platform product"""
        inconsistencies = []
        
        # Compare key fields
        if db_product.title != platform_product.get('title', ''):
            inconsistencies.append({
                'field': 'title',
                'database_value': db_product.title,
                'platform_value': platform_product.get('title', '')
            })
            
        if str(db_product.price) != str(platform_product.get('price', '0.00')):
            inconsistencies.append({
                'field': 'price',
                'database_value': str(db_product.price),
                'platform_value': str(platform_product.get('price', '0.00'))
            })
            
        if db_product.inventory_quantity != platform_product.get('inventory_quantity', 0):
            inconsistencies.append({
                'field': 'inventory_quantity',
                'database_value': db_product.inventory_quantity,
                'platform_value': platform_product.get('inventory_quantity', 0)
            })
            
        if inconsistencies:
            return {
                'product_id': db_product.external_id,
                'inconsistencies': inconsistencies,
                'detected_at': datetime.now(timezone.utc).isoformat()
            }
            
        return None
        
    async def _transform_products_for_platform(
        self, 
        products: List[Dict[str, Any]], 
        target_platform: str, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Transform products for target platform"""
        # This would implement platform-specific transformations
        # For now, return products as-is
        return products
        
    async def _migrate_products_to_platform(
        self, 
        tenant_id: str, 
        target_platform: str, 
        products: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Migrate products to target platform"""
        # This would implement actual platform migration
        # For now, return mock results
        return {
            'products_migrated': len(products),
            'successful_migrations': len(products),
            'failed_migrations': 0
        }
        
    async def _load_sync_schedules(self) -> None:
        """Load synchronization schedules"""
        # This would typically load from database
        pass
        
    def _calculate_next_run(self, cron_expression: str) -> datetime:
        """Calculate next run time from cron expression"""
        # This would implement cron parsing
        # For now, return next hour
        return datetime.now(timezone.utc) + timedelta(hours=1)
        
    def _calculate_sync_progress(self, job: SyncJob) -> float:
        """Calculate sync progress percentage"""
        if job.status == SyncStatus.COMPLETED:
            return 100.0
        elif job.status == SyncStatus.FAILED:
            return 0.0
        elif job.status == SyncStatus.IN_PROGRESS:
            # Estimate progress based on time elapsed
            elapsed = datetime.now(timezone.utc) - job.started_at
            estimated_duration = 600  # 10 minutes estimate
            progress = min(90.0, (elapsed.total_seconds() / estimated_duration) * 100)
            return progress
        else:
            return 0.0
            
    async def _get_last_sync_timestamp(self, tenant_id: str, platform: str) -> Optional[datetime]:
        """Get last sync timestamp for tenant and platform"""
        # This would typically query a sync_logs table
        return None
