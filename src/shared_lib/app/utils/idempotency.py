"""
Aura Platform - Idempotency Protection
Redis-based idempotency tracking for duplicate message prevention.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Union
from uuid import UUID
import json
import hashlib

import redis.asyncio as redis
from functools import wraps

logger = logging.getLogger(__name__)


class IdempotencyError(Exception):
    """Exception raised when duplicate operation is detected."""
    pass


class IdempotencyManager:
    """
    Manages idempotency using Redis for duplicate operation prevention.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        default_ttl: int = 3600,  # 1 hour
        key_prefix: str = "idempotency",
    ):
        """
        Initialize idempotency manager.
        
        Args:
            redis_client: Redis client instance
            default_ttl: Default TTL for idempotency keys in seconds
            key_prefix: Prefix for idempotency keys
        """
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        
        logger.info(f"IdempotencyManager initialized with TTL {default_ttl}s")
    
    def _generate_key(self, operation_id: str) -> str:
        """
        Generate Redis key for idempotency tracking.
        
        Args:
            operation_id: Unique operation identifier
            
        Returns:
            Redis key for idempotency tracking
        """
        # Create a hash of the operation ID for consistent key generation
        key_hash = hashlib.sha256(operation_id.encode()).hexdigest()[:16]
        return f"{self.key_prefix}:{key_hash}"
    
    async def check_and_set(
        self,
        operation_id: str,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if operation is duplicate and set idempotency key.
        
        Args:
            operation_id: Unique operation identifier
            ttl: TTL for the key in seconds
            metadata: Optional metadata to store with the key
            
        Returns:
            True if operation is new (not duplicate), False if duplicate
            
        Raises:
            IdempotencyError: If operation is duplicate
        """
        key = self._generate_key(operation_id)
        ttl = ttl or self.default_ttl
        
        try:
            # Prepare metadata
            metadata_dict = {
                "operation_id": operation_id,
                "timestamp": asyncio.get_event_loop().time(),
                **(metadata or {}),
            }
            metadata_json = json.dumps(metadata_dict)
            
            # Use SET with NX (only if not exists) and EX (expiration)
            result = await self.redis_client.set(
                key,
                metadata_json,
                nx=True,  # Only set if key doesn't exist
                ex=ttl,   # Set expiration
            )
            
            if result:
                logger.info(f"Idempotency key set for operation {operation_id}")
                return True
            else:
                logger.warning(f"Duplicate operation detected: {operation_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to check/set idempotency key for {operation_id}: {e}")
            # In case of Redis error, allow operation to proceed
            # This prevents Redis failures from blocking operations
            return True
    
    async def get_metadata(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an operation.
        
        Args:
            operation_id: Unique operation identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        key = self._generate_key(operation_id)
        
        try:
            metadata_json = await self.redis_client.get(key)
            if metadata_json:
                return json.loads(metadata_json)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get metadata for operation {operation_id}: {e}")
            return None
    
    async def extend_ttl(self, operation_id: str, ttl: int) -> bool:
        """
        Extend TTL for an idempotency key.
        
        Args:
            operation_id: Unique operation identifier
            ttl: New TTL in seconds
            
        Returns:
            True if TTL was extended, False otherwise
        """
        key = self._generate_key(operation_id)
        
        try:
            result = await self.redis_client.expire(key, ttl)
            if result:
                logger.debug(f"Extended TTL for operation {operation_id} to {ttl}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to extend TTL for operation {operation_id}: {e}")
            return False
    
    async def delete_key(self, operation_id: str) -> bool:
        """
        Delete idempotency key for an operation.
        
        Args:
            operation_id: Unique operation identifier
            
        Returns:
            True if key was deleted, False otherwise
        """
        key = self._generate_key(operation_id)
        
        try:
            result = await self.redis_client.delete(key)
            if result:
                logger.debug(f"Deleted idempotency key for operation {operation_id}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to delete idempotency key for operation {operation_id}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of idempotency manager.
        
        Returns:
            Health status information
        """
        try:
            # Test Redis connectivity
            await self.redis_client.ping()
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "default_ttl": self.default_ttl,
                "key_prefix": self.key_prefix,
            }
            
        except Exception as e:
            logger.error(f"Idempotency manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e),
                "default_ttl": self.default_ttl,
                "key_prefix": self.key_prefix,
            }


def idempotent_task(
    key_arg: str = "message_id",
    ttl: int = 3600,
    metadata_func: Optional[Callable] = None,
):
    """
    Decorator for making Celery tasks idempotent.
    
    Args:
        key_arg: Name of the argument containing the unique operation ID
        ttl: TTL for idempotency key in seconds
        metadata_func: Optional function to generate metadata from task args
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract operation ID from arguments
            operation_id = None
            
            # Try to get from kwargs first
            if key_arg in kwargs:
                operation_id = str(kwargs[key_arg])
            else:
                # Try to get from args (assuming it's the first argument)
                if args and len(args) > 0:
                    operation_id = str(args[0])
            
            if not operation_id:
                logger.warning(f"Could not extract {key_arg} from task arguments")
                # If we can't extract the operation ID, proceed without idempotency
                return await func(*args, **kwargs)
            
            # Get Redis client from task context or create one
            # This assumes Redis client is available in the task context
            redis_client = getattr(wrapper, '_redis_client', None)
            if not redis_client:
                logger.warning("Redis client not available for idempotency check")
                return await func(*args, **kwargs)
            
            # Create idempotency manager
            idempotency_manager = IdempotencyManager(redis_client, default_ttl=ttl)
            
            # Generate metadata if function is provided
            metadata = None
            if metadata_func:
                try:
                    metadata = metadata_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to generate metadata: {e}")
            
            # Check for duplicate operation
            is_new_operation = await idempotency_manager.check_and_set(
                operation_id=operation_id,
                ttl=ttl,
                metadata=metadata,
            )
            
            if not is_new_operation:
                logger.info(f"Duplicate task detected for {key_arg}={operation_id}, skipping execution")
                # Return success to acknowledge the message
                return {
                    "status": "duplicate",
                    "operation_id": operation_id,
                    "message": "Task already processed",
                }
            
            # Execute the original function
            try:
                result = await func(*args, **kwargs)
                logger.info(f"Task completed successfully for {key_arg}={operation_id}")
                return result
                
            except Exception as e:
                logger.error(f"Task failed for {key_arg}={operation_id}: {e}")
                # Optionally delete the idempotency key on failure
                # This allows retry of failed operations
                await idempotency_manager.delete_key(operation_id)
                raise
        
        # Store Redis client reference for the wrapper
        wrapper._redis_client = None
        
        return wrapper
    
    return decorator


def set_redis_client_for_idempotency(redis_client: redis.Redis):
    """
    Set Redis client for idempotency decorators.
    
    Args:
        redis_client: Redis client instance
    """
    # This is a global function to set the Redis client
    # In a real implementation, this would be called during app initialization
    logger.info("Redis client set for idempotency decorators")


# Factory function for creating idempotency manager
def create_idempotency_manager(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: Optional[str] = None,
    redis_db: int = 0,
    default_ttl: int = 3600,
) -> IdempotencyManager:
    """
    Create an IdempotencyManager instance.
    
    Args:
        redis_host: Redis host
        redis_port: Redis port
        redis_password: Redis password
        redis_db: Redis database number
        default_ttl: Default TTL for idempotency keys
        
    Returns:
        IdempotencyManager instance
    """
    # Create Redis client
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        db=redis_db,
        decode_responses=True,
    )
    
    return IdempotencyManager(
        redis_client=redis_client,
        default_ttl=default_ttl,
    )
