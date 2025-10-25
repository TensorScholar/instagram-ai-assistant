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

# Create Celery application
celery_app = Celery(
    "intelligence_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.message_processing"],
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
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,  # 10 minutes
)

# Configure task routes
celery_app.conf.task_routes = {
    "app.tasks.message_processing.*": {"queue": "intelligence_queue"},
}

# Configure queues
celery_app.conf.task_default_queue = "intelligence_queue"
celery_app.conf.task_queues = {
    "intelligence_queue": {
        "exchange": "intelligence_exchange",
        "routing_key": "intelligence",
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
