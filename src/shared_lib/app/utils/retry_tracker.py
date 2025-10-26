"""
Aura Platform - Retry Tracker
Redis-based retry tracking for poison pill detection and DLQ routing.
"""

import logging
from typing import Optional
from uuid import UUID
import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RetryTracker:
    """
    Redis-based retry tracker for poison pill detection and DLQ routing.
    
    This class tracks the number of retries for each message ID and determines
    when a message should be sent to the Dead-Letter Queue (DLQ).
    """
    
    def __init__(
        self,
        redis_client: Redis,
        max_retries: int = 3,
        retry_ttl: int = 3600,  # 1 hour TTL for retry counters
    ):
        """
        Initialize retry tracker.
        
        Args:
            redis_client: Redis client instance
            max_retries: Maximum number of retries before DLQ routing
            retry_ttl: Time-to-live for retry counters in seconds
        """
        self.redis_client = redis_client
        self.max_retries = max_retries
        self.retry_ttl = retry_ttl
        logger.info(f"RetryTracker initialized with max_retries={max_retries}, ttl={retry_ttl}s")
    
    async def increment_and_check(self, message_id: str) -> bool:
        """
        Increment retry counter for a message and check if it should go to DLQ.
        
        Args:
            message_id: Unique message identifier
            
        Returns:
            True if message should be sent to DLQ, False otherwise
        """
        try:
            # Convert UUID to string if needed
            if isinstance(message_id, UUID):
                message_id = str(message_id)
            
            # Redis key for retry counter
            retry_key = f"retry_count:{message_id}"
            
            # Increment counter and get current count
            current_count = await self.redis_client.incr(retry_key)
            
            # Set TTL on first increment
            if current_count == 1:
                await self.redis_client.expire(retry_key, self.retry_ttl)
            
            # Check if max retries exceeded
            should_send_to_dlq = current_count >= self.max_retries
            
            if should_send_to_dlq:
                logger.warning(
                    f"Message {message_id} exceeded max retries ({current_count}/{self.max_retries}). "
                    "Routing to DLQ."
                )
            else:
                logger.debug(
                    f"Message {message_id} retry count: {current_count}/{self.max_retries}"
                )
            
            return should_send_to_dlq
            
        except Exception as e:
            logger.error(f"Error tracking retries for message {message_id}: {e}")
            # On error, assume it's not a poison pill to avoid false positives
            return False
    
    async def get_retry_count(self, message_id: str) -> int:
        """
        Get current retry count for a message.
        
        Args:
            message_id: Unique message identifier
            
        Returns:
            Current retry count
        """
        try:
            if isinstance(message_id, UUID):
                message_id = str(message_id)
            
            retry_key = f"retry_count:{message_id}"
            count = await self.redis_client.get(retry_key)
            return int(count) if count else 0
            
        except Exception as e:
            logger.error(f"Error getting retry count for message {message_id}: {e}")
            return 0
    
    async def reset_retry_count(self, message_id: str) -> None:
        """
        Reset retry count for a message.
        
        Args:
            message_id: Unique message identifier
        """
        try:
            if isinstance(message_id, UUID):
                message_id = str(message_id)
            
            retry_key = f"retry_count:{message_id}"
            await self.redis_client.delete(retry_key)
            logger.debug(f"Reset retry count for message {message_id}")
            
        except Exception as e:
            logger.error(f"Error resetting retry count for message {message_id}: {e}")
    
    async def cleanup_expired_retries(self) -> int:
        """
        Clean up expired retry counters.
        
        Returns:
            Number of expired counters removed
        """
        try:
            # Get all retry counter keys
            retry_keys = await self.redis_client.keys("retry_count:*")
            
            if not retry_keys:
                return 0
            
            # Check TTL for each key
            expired_keys = []
            for key in retry_keys:
                ttl = await self.redis_client.ttl(key)
                if ttl == -1:  # No TTL set
                    await self.redis_client.expire(key, self.retry_ttl)
                elif ttl == -2:  # Key doesn't exist
                    expired_keys.append(key)
            
            # Remove expired keys
            if expired_keys:
                await self.redis_client.delete(*expired_keys)
                logger.info(f"Cleaned up {len(expired_keys)} expired retry counters")
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired retries: {e}")
            return 0


class PoisonPillDetector:
    """
    Detects poison pill messages that should be immediately routed to DLQ.
    """
    
    def __init__(self):
        """Initialize poison pill detector."""
        # Critical, non-transient errors that indicate poison pills
        self.poison_patterns = [
            "json.JSONDecodeError",
            "UnicodeDecodeError", 
            "AttributeError.*NoneType",
            "KeyError.*required",
            "ValueError.*invalid",
            "TypeError.*unexpected",
        ]
        logger.info("PoisonPillDetector initialized")
    
    def is_poison_pill(self, exception: Exception) -> bool:
        """
        Detect if an exception indicates a poison pill message.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if the exception indicates a poison pill
        """
        try:
            exception_str = str(exception)
            exception_type = type(exception).__name__
            
            # Check exception type
            if exception_type in ["JSONDecodeError", "UnicodeDecodeError", "AttributeError", "KeyError", "ValueError", "TypeError"]:
                logger.warning(f"Poison pill detected: {exception_type} - {exception_str}")
                return True
            
            # Check exception message patterns
            for pattern in self.poison_patterns:
                if pattern.lower() in exception_str.lower():
                    logger.warning(f"Poison pill pattern detected: {pattern} in {exception_str}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting poison pill: {e}")
            return False
    
    def get_poison_reason(self, exception: Exception) -> str:
        """
        Get a human-readable reason for why a message is considered a poison pill.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            Human-readable poison pill reason
        """
        try:
            exception_type = type(exception).__name__
            exception_str = str(exception)
            
            if exception_type == "JSONDecodeError":
                return f"Malformed JSON: {exception_str}"
            elif exception_type == "UnicodeDecodeError":
                return f"Invalid encoding: {exception_str}"
            elif exception_type == "AttributeError":
                return f"Missing required attribute: {exception_str}"
            elif exception_type == "KeyError":
                return f"Missing required key: {exception_str}"
            elif exception_type == "ValueError":
                return f"Invalid value: {exception_str}"
            elif exception_type == "TypeError":
                return f"Type mismatch: {exception_str}"
            else:
                return f"Poison pill pattern detected: {exception_str}"
                
        except Exception as e:
            logger.error(f"Error getting poison reason: {e}")
            return f"Unknown poison pill: {exception}"
