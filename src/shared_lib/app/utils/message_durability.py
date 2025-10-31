"""
Aura Platform - Message Durability Manager
RabbitMQ message durability with publisher confirms and dead-letter queue.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Union
from uuid import UUID
import json

try:
    import pika  # type: ignore
    from pika.exchange_type import ExchangeType  # type: ignore
except Exception:
    pika = None  # type: ignore
    ExchangeType = None  # type: ignore
try:
    from celery import Celery  # type: ignore
except Exception:
    Celery = None  # type: ignore

logger = logging.getLogger(__name__)


class MessageDurabilityManager:
    """
    Manages message durability with publisher confirms and dead-letter queue.
    """
    
    def __init__(
        self,
        rabbitmq_host: str = "localhost",
        rabbitmq_port: int = 5672,
        rabbitmq_username: str = "guest",
        rabbitmq_password: str = "guest",
        rabbitmq_vhost: str = "/",
    ):
        """
        Initialize message durability manager.
        
        Args:
            rabbitmq_host: RabbitMQ host
            rabbitmq_port: RabbitMQ port
            rabbitmq_username: RabbitMQ username
            rabbitmq_password: RabbitMQ password
            rabbitmq_vhost: RabbitMQ virtual host
        """
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_username = rabbitmq_username
        self.rabbitmq_password = rabbitmq_password
        self.rabbitmq_vhost = rabbitmq_vhost
        
        self.connection = None
        self.channel = None
        
        logger.info(f"MessageDurabilityManager initialized for {rabbitmq_host}:{rabbitmq_port}")
    
    async def connect(self) -> None:
        """
        Establish connection to RabbitMQ with durability settings.
        """
        try:
            if pika is None:
                raise RuntimeError("pika SDK not available")
            # Create connection parameters
            credentials = pika.PlainCredentials(
                username=self.rabbitmq_username,
                password=self.rabbitmq_password,
            )
            
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                virtual_host=self.rabbitmq_vhost,
                credentials=credentials,
                heartbeat=600,  # 10 minutes
                blocked_connection_timeout=300,  # 5 minutes
            )
            
            # Establish connection
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Enable publisher confirms
            if hasattr(self.channel, "confirm_delivery"):
                self.channel.confirm_delivery()
            
            # Declare dead-letter exchange
            if ExchangeType is None:
                raise RuntimeError("pika ExchangeType not available")
            self.channel.exchange_declare(
                exchange="dlx",
                exchange_type=ExchangeType.direct,
                durable=True,
            )
            
            # Declare dead-letter queue
            self.channel.queue_declare(
                queue="dead_letter_queue",
                durable=True,
            )
            
            # Bind dead-letter queue to dead-letter exchange
            self.channel.queue_bind(
                exchange="dlx",
                queue="dead_letter_queue",
                routing_key="failed",
            )
            
            logger.info("Connected to RabbitMQ with durability settings")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def declare_durable_queue(
        self,
        queue_name: str,
        max_retries: int = 3,
        retry_delay: int = 5000,  # 5 seconds
    ) -> None:
        """
        Declare a durable queue with dead-letter configuration.
        
        Args:
            queue_name: Name of the queue
            max_retries: Maximum number of retries before sending to DLQ
            retry_delay: Delay between retries in milliseconds
        """
        try:
            # Declare main queue with dead-letter configuration
            self.channel.queue_declare(
                queue=queue_name,
                durable=True,  # Queue survives broker restarts
                arguments={
                    "x-dead-letter-exchange": "dlx",
                    "x-dead-letter-routing-key": "failed",
                    "x-message-ttl": retry_delay * max_retries,  # TTL for retries
                    "x-max-retries": max_retries,
                },
            )
            
            logger.info(f"Declared durable queue {queue_name} with DLQ configuration")
            
        except Exception as e:
            logger.error(f"Failed to declare durable queue {queue_name}: {e}")
            raise
    
    async def publish_message(
        self,
        exchange: str,
        routing_key: str,
        message: Dict[str, Any],
        queue_name: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """
        Publish a message with durability guarantees.
        
        Args:
            exchange: Exchange name
            routing_key: Routing key
            message: Message payload
            queue_name: Optional queue name for direct publishing
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            True if message was published successfully, False otherwise
        """
        try:
            # Prepare message properties
            properties = pika.BasicProperties(
                delivery_mode=2,  # Persistent message
                correlation_id=correlation_id or str(UUID()),
                timestamp=int(asyncio.get_event_loop().time()),
                content_type="application/json",
            )
            
            # Serialize message
            message_body = json.dumps(message).encode('utf-8')
            
            # Publish message
            if queue_name:
                # Direct queue publishing
                self.channel.basic_publish(
                    exchange="",
                    routing_key=queue_name,
                    body=message_body,
                    properties=properties,
                    mandatory=True,  # Ensure message is routed
                )
            else:
                # Exchange publishing
                self.channel.basic_publish(
                    exchange=exchange,
                    routing_key=routing_key,
                    body=message_body,
                    properties=properties,
                    mandatory=True,  # Ensure message is routed
                )
            
            logger.info(f"Published message to {exchange or queue_name} with routing key {routing_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def publish_with_confirm(
        self,
        exchange: str,
        routing_key: str,
        message: Dict[str, Any],
        queue_name: Optional[str] = None,
        correlation_id: Optional[str] = None,
        timeout: int = 30,
    ) -> bool:
        """
        Publish a message with publisher confirm.
        
        Args:
            exchange: Exchange name
            routing_key: Routing key
            message: Message payload
            queue_name: Optional queue name for direct publishing
            correlation_id: Optional correlation ID for tracing
            timeout: Timeout for confirmation in seconds
            
        Returns:
            True if message was confirmed, False otherwise
        """
        try:
            # Prepare message properties
            properties = pika.BasicProperties(
                delivery_mode=2,  # Persistent message
                correlation_id=correlation_id or str(UUID()),
                timestamp=int(asyncio.get_event_loop().time()),
                content_type="application/json",
            )
            
            # Serialize message
            message_body = json.dumps(message).encode('utf-8')
            
            # Publish with confirmation
            if queue_name:
                # Direct queue publishing
                confirmed = self.channel.basic_publish(
                    exchange="",
                    routing_key=queue_name,
                    body=message_body,
                    properties=properties,
                    mandatory=True,
                )
            else:
                # Exchange publishing
                confirmed = self.channel.basic_publish(
                    exchange=exchange,
                    routing_key=routing_key,
                    body=message_body,
                    properties=properties,
                    mandatory=True,
                )
            
            if confirmed:
                logger.info(f"Message confirmed for {exchange or queue_name} with routing key {routing_key}")
                return True
            else:
                logger.warning(f"Message not confirmed for {exchange or queue_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to publish message with confirm: {e}")
            return False
    
    async def setup_celery_durability(self, celery_app: Any) -> None:
        """
        Configure Celery for message durability.
        
        Args:
            celery_app: Celery application instance
        """
        try:
            if Celery is None:
                logger.warning("Celery not available; skipping Celery durability configuration")
                return
            # Configure Celery for durability
            celery_app.conf.update(
                # Broker settings
                broker_transport_options={
                    'visibility_timeout': 3600,  # 1 hour
                    'fanout_prefix': True,
                    'fanout_patterns': True,
                },
                
                # Task settings
                task_acks_late=True,  # Acknowledge after task completion
                task_reject_on_worker_lost=True,  # Reject tasks on worker loss
                task_default_retry_delay=60,  # 1 minute retry delay
                task_max_retries=3,  # Maximum retries
                
                # Worker settings
                worker_prefetch_multiplier=1,  # Process one task at a time
                worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
                
                # Result backend settings
                result_backend_transport_options={
                    'master_name': 'mymaster',
                },
                
                # Serialization
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
                
                # Timezone
                timezone='UTC',
                enable_utc=True,
            )
            
            logger.info("Configured Celery for message durability")
            
        except Exception as e:
            logger.error(f"Failed to configure Celery durability: {e}")
            raise
    
    async def get_queue_info(self, queue_name: str) -> Dict[str, Any]:
        """
        Get information about a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Queue information dictionary
        """
        try:
            # Declare queue to get info
            method = self.channel.queue_declare(
                queue=queue_name,
                passive=True,  # Only check if queue exists
            )
            
            return {
                "queue_name": queue_name,
                "message_count": method.method.message_count,
                "consumer_count": method.method.consumer_count,
                "durable": True,
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue info for {queue_name}: {e}")
            return {
                "queue_name": queue_name,
                "error": str(e),
            }
    
    async def purge_queue(self, queue_name: str) -> int:
        """
        Purge all messages from a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Number of messages purged
        """
        try:
            method = self.channel.queue_purge(queue=queue_name)
            purged_count = method.method.message_count
            
            logger.info(f"Purged {purged_count} messages from queue {queue_name}")
            return purged_count
            
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {e}")
            return 0
    
    async def close(self) -> None:
        """
        Close connection to RabbitMQ.
        """
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            
            logger.info("Closed RabbitMQ connection")
            
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of message durability manager.
        
        Returns:
            Health status information
        """
        try:
            if not self.connection or self.connection.is_closed:
                return {
                    "status": "unhealthy",
                    "error": "Not connected to RabbitMQ",
                }
            
            # Test connection
            self.connection.process_data_events()
            
            return {
                "status": "healthy",
                "rabbitmq_host": self.rabbitmq_host,
                "rabbitmq_port": self.rabbitmq_port,
                "connected": True,
            }
            
        except Exception as e:
            logger.error(f"Message durability manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "rabbitmq_host": self.rabbitmq_host,
                "rabbitmq_port": self.rabbitmq_port,
            }


# Factory function for creating message durability manager
def create_message_durability_manager(
    rabbitmq_host: str = "localhost",
    rabbitmq_port: int = 5672,
    rabbitmq_username: str = "guest",
    rabbitmq_password: str = "guest",
    rabbitmq_vhost: str = "/",
) -> MessageDurabilityManager:
    """
    Create a MessageDurabilityManager instance.
    
    Args:
        rabbitmq_host: RabbitMQ host
        rabbitmq_port: RabbitMQ port
        rabbitmq_username: RabbitMQ username
        rabbitmq_password: RabbitMQ password
        rabbitmq_vhost: RabbitMQ virtual host
        
    Returns:
        MessageDurabilityManager instance
    """
    return MessageDurabilityManager(
        rabbitmq_host=rabbitmq_host,
        rabbitmq_port=rabbitmq_port,
        rabbitmq_username=rabbitmq_username,
        rabbitmq_password=rabbitmq_password,
        rabbitmq_vhost=rabbitmq_vhost,
    )
