"""
Aura Platform - Enhanced Celery Task Decorators
Enhanced task decorators with poison pill detection and DLQ routing.
"""

import logging
from functools import wraps
from typing import Any, Callable, Awaitable, Optional
from uuid import UUID
import json

from celery import Task
from celery.exceptions import Retry, Ignore

from ..utils.retry_tracker import RetryTracker, PoisonPillDetector
from ..utils.idempotency import get_redis_client

logger = logging.getLogger(__name__)


def resilient_task(
    bind: bool = True,
    max_retries: int = 3,
    default_retry_delay: int = 60,
    poison_pill_detection: bool = True,
    dlq_routing: bool = True,
    **kwargs: Any
):
    """
    Enhanced Celery task decorator with poison pill detection and DLQ routing.
    
    Args:
        bind: Whether to bind the task instance
        max_retries: Maximum number of retries
        default_retry_delay: Default delay between retries
        poison_pill_detection: Enable poison pill detection
        dlq_routing: Enable automatic DLQ routing
        **kwargs: Additional Celery task configuration
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get task instance if bound
            task_instance = args[0] if args and isinstance(args[0], Task) else None
            task_args = args[1:] if task_instance else args
            
            # Extract message_id from task arguments
            message_id = None
            if 'message_id' in kwargs:
                message_id = kwargs['message_id']
            elif task_args and len(task_args) > 0:
                # Try to extract from first argument if it's a dict
                if isinstance(task_args[0], dict) and 'message_id' in task_args[0]:
                    message_id = task_args[0]['message_id']
            
            if not message_id:
                logger.warning("No message_id found in task arguments. Poison pill detection disabled.")
                return await func(*args, **kwargs)
            
            # Initialize components
            retry_tracker = None
            poison_detector = None
            
            if poison_pill_detection or dlq_routing:
                try:
                    redis_client = get_redis_client()
                    retry_tracker = RetryTracker(redis_client, max_retries=max_retries)
                    poison_detector = PoisonPillDetector()
                except Exception as e:
                    logger.error(f"Failed to initialize retry tracking: {e}")
                    # Continue without poison pill detection
            
            try:
                # Execute the task
                result = await func(*args, **kwargs)
                
                # Reset retry count on success
                if retry_tracker and message_id:
                    await retry_tracker.reset_retry_count(message_id)
                
                return result
                
            except Exception as e:
                logger.error(f"Task {func.__name__} failed for message {message_id}: {e}")
                
                # Check if this is a poison pill
                is_poison = False
                poison_reason = None
                
                if poison_detector and poison_pill_detection:
                    is_poison = poison_detector.is_poison_pill(e)
                    if is_poison:
                        poison_reason = poison_detector.get_poison_reason(e)
                        logger.critical(f"Poison pill detected for message {message_id}: {poison_reason}")
                
                # Handle retry tracking and DLQ routing
                if retry_tracker and dlq_routing and message_id:
                    try:
                        # Use async variant if available
                        if hasattr(retry_tracker, 'aincrement_and_check'):
                            should_send_to_dlq = await retry_tracker.aincrement_and_check(message_id)
                        else:
                            should_send_to_dlq = retry_tracker.increment_and_check(message_id)
                        
                        if should_send_to_dlq or is_poison:
                            # Route to DLQ
                            dlq_reason = "poison_pill" if is_poison else "max_retries_exceeded"
                            logger.critical(f"Routing message {message_id} to DLQ: {dlq_reason}")
                            
                            # Publish to DLQ (this would be implemented based on your DLQ setup)
                            await _publish_to_dlq(message_id, dlq_reason, str(e))
                            
                            # Don't re-raise the exception to prevent further Celery retries
                            return {
                                "status": "dlq_routed",
                                "message_id": message_id,
                                "reason": dlq_reason,
                                "error": str(e)
                            }
                            
                    except Exception as tracking_error:
                        logger.error(f"Error in retry tracking for message {message_id}: {tracking_error}")
                
                # If not routed to DLQ, handle normal retry logic
                if task_instance and task_instance.request.retries < max_retries:
                    logger.info(f"Retrying task {func.__name__} for message {message_id} (attempt {task_instance.request.retries + 1}/{max_retries})")
                    raise task_instance.retry(
                        exc=e,
                        countdown=default_retry_delay,
                        max_retries=max_retries
                    )
                else:
                    logger.error(f"Task {func.__name__} failed permanently for message {message_id} after {max_retries} retries")
                    raise
        
        # Apply Celery task decorator
        # Note: This would be imported from the actual Celery app in real usage
        # For now, we'll return the wrapper function
        return wrapper
    
    return decorator


async def _publish_to_dlq(message_id: str, reason: str, error_details: str) -> None:
    """
    Publish a message to the Dead-Letter Queue.
    
    Args:
        message_id: The message ID
        reason: Reason for DLQ routing
        error_details: Error details
    """
    try:
        # This would be implemented based on your DLQ setup
        # For now, we'll just log the DLQ routing
        dlq_message = {
            "message_id": message_id,
            "reason": reason,
            "error_details": error_details,
            "timestamp": str(datetime.now()),
            "original_queue": "intelligence_queue"
        }
        
        logger.critical(f"DLQ Message: {json.dumps(dlq_message)}")
        
        # In a real implementation, you would publish this to your DLQ
        # For example, using RabbitMQ DLQ or a dedicated DLQ service
        
    except Exception as e:
        logger.error(f"Failed to publish message {message_id} to DLQ: {e}")


# Import datetime for DLQ message
from datetime import datetime
