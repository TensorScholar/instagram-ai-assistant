"""
Aura Platform - Intelligence Worker Celery Application
Celery application configuration for the Intelligence Worker service.
"""

import logging
from typing import Any, Dict

from celery import Celery
from celery.signals import worker_ready, worker_shutdown

from .config import settings

logger = logging.getLogger(__name__)

# Construct Redis URLs from discrete environment variables
def get_redis_url(host: str, port: int, password: str, db: int = 0) -> str:
    """Construct Redis URL from discrete components."""
    return f"redis://:{password}@{host}:{port}/{db}"

# Create Celery application with Redis backend
celery_app = Celery(
    "intelligence_worker",
    broker=get_redis_url(settings.redis_host, settings.redis_port, settings.redis_password),
    backend=get_redis_url(settings.redis_host, settings.redis_port, settings.redis_password, settings.redis_db),
    include=[
        "app.tasks.message_processing",
        "app.tasks.resilient_ai_processing",
    ],
)

# Configure Celery
celery_app.conf.update(
    task_serializer=settings.celery_task_serializer,
    result_serializer=settings.celery_result_serializer,
    accept_content=settings.celery_accept_content,
    timezone=settings.celery_timezone,
    enable_utc=settings.celery_enable_utc,
    worker_prefetch_multiplier=settings.worker_prefetch_multiplier,
    worker_max_tasks_per_child=settings.worker_max_tasks_per_child,
    task_always_eager=False,  # Set to True for testing
    task_eager_propagates=True,
    task_ignore_result=False,
    result_expires=3600,  # 1 hour
    task_soft_time_limit=45,   # 45 seconds soft limit
    task_time_limit=60,        # 60 seconds hard limit - kills hanging tasks
    # Durability settings
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,  # Reject tasks on worker loss
    task_default_retry_delay=60,  # 1 minute retry delay
    task_max_retries=3,  # Maximum retries
    # Broker settings
    broker_transport_options={
        'visibility_timeout': 3600,  # 1 hour
        'fanout_prefix': True,
        'fanout_patterns': True,
    },
)

# Configure task routes with separate queues
celery_app.conf.task_routes = {
    "app.tasks.message_processing.*": {"queue": "realtime_queue"},
    "app.tasks.resilient_ai_processing.*": {"queue": "realtime_queue"},
    "app.tasks.bulk_ingestion.*": {"queue": "bulk_queue"},
}

# Configure separate queues for different priorities
celery_app.conf.task_default_queue = "realtime_queue"
celery_app.conf.task_queues = {
    "realtime_queue": {
        "exchange": "aura_events",
        "routing_key": "realtime",
        "queue_arguments": {
            "x-max-length": 1000,  # Limit queue size
            "x-message-ttl": 300000,  # 5 minute TTL
            "x-max-retries": 3,
            "x-dead-letter-exchange": "dlx",
            "x-dead-letter-routing-key": "failed",
        }
    },
    "bulk_queue": {
        "exchange": "aura_events", 
        "routing_key": "bulk",
        "queue_arguments": {
            "x-max-length": 5000,  # Larger limit for bulk operations
            "x-message-ttl": 1800000,  # 30 minute TTL
            "x-max-retries": 5,
            "x-dead-letter-exchange": "dlx",
            "x-dead-letter-routing-key": "failed",
        }
    }
}


@worker_ready.connect
def worker_ready_handler(sender: Any, **kwargs: Any) -> None:
    """
    Handle worker ready signal.
    
    Args:
        sender: The worker instance
        **kwargs: Additional arguments
    """
    logger.info("Intelligence Worker is ready to process tasks")


@worker_shutdown.connect
def worker_shutdown_handler(sender: Any, **kwargs: Any) -> None:
    """
    Handle worker shutdown signal.
    
    Args:
        sender: The worker instance
        **kwargs: Additional arguments
    """
    logger.info("Intelligence Worker is shutting down")


# Task decorators for different types of processing
def intelligence_task(**kwargs: Any):
    """
    Decorator for intelligence processing tasks.
    
    Args:
        **kwargs: Additional task configuration
        
    Returns:
        The decorated task function
    """
    default_kwargs = {
        "bind": True,
        "autoretry_for": (Exception,),
        "retry_kwargs": {"max_retries": 3, "countdown": 60},
        "retry_backoff": True,
        "retry_jitter": True,
    }
    default_kwargs.update(kwargs)
    
    return celery_app.task(**default_kwargs)


def message_task(**kwargs: Any):
    """
    Decorator for message processing tasks.
    
    Args:
        **kwargs: Additional task configuration
        
    Returns:
        The decorated task function
    """
    default_kwargs = {
        "bind": True,
        "autoretry_for": (Exception,),
        "retry_kwargs": {"max_retries": 3, "countdown": 30},
        "retry_backoff": True,
        "retry_jitter": True,
    }
    default_kwargs.update(kwargs)
    
    return celery_app.task(**default_kwargs)


# Health check task
@celery_app.task(bind=True)
def health_check_task(self) -> Dict[str, Any]:
    """
    Health check task for the Intelligence Worker.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "service": "intelligence_worker",
        "version": settings.app_version,
        "environment": settings.environment,
        "worker_id": self.request.id,
    }


if __name__ == "__main__":
    # Start the Celery worker
    celery_app.start()
