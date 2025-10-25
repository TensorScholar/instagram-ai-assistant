"""
Aura Platform - RabbitMQ Integration
RabbitMQ integration for event-driven communication between services.
"""

import json
import logging
from typing import Any, Callable, Dict, Optional
from uuid import UUID

import pika
from pika.adapters.blocking_connection import BlockingConnection
from pika.connection import ConnectionParameters
from pika.exchange_type import ExchangeType

from shared_lib.app.schemas.events import BaseEvent, EventType, get_event_class

logger = logging.getLogger(__name__)


class RabbitMQManager:
    """RabbitMQ manager for event publishing and consuming."""
    
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        vhost: str = "/",
    ):
        """
        Initialize RabbitMQ manager.
        
        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            username: RabbitMQ username
            password: RabbitMQ password
            vhost: RabbitMQ virtual host
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.vhost = vhost
        
        self.connection: Optional[BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
        
        # Exchange and queue configurations
        self.exchange_name = "aura_events"
        self.queue_name = "intelligence_queue"
        self.routing_key = "intelligence"
    
    def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.vhost,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            
            self.connection = BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare exchange
            self.channel.exchange_declare(
                exchange=self.exchange_name,
                exchange_type=ExchangeType.topic,
                durable=True,
            )
            
            # Declare queue
            self.channel.queue_declare(
                queue=self.queue_name,
                durable=True,
            )
            
            # Bind queue to exchange
            self.channel.queue_bind(
                exchange=self.exchange_name,
                queue=self.queue_name,
                routing_key=self.routing_key,
            )
            
            logger.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")
    
    def publish_event(self, event: BaseEvent) -> bool:
        """
        Publish an event to RabbitMQ.
        
        Args:
            event: The event to publish
            
        Returns:
            True if published successfully, False otherwise
        """
        try:
            if not self.channel or self.channel.is_closed:
                self.connect()
            
            # Serialize event
            event_data = event.dict()
            message_body = json.dumps(event_data, default=str)
            
            # Publish message
            self.channel.basic_publish(
                exchange=self.exchange_name,
                routing_key=self.routing_key,
                body=message_body,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type="application/json",
                    headers={
                        "event_type": event.event_type,
                        "tenant_id": str(event.tenant_id),
                        "correlation_id": str(event.correlation_id),
                    },
                ),
            )
            
            logger.info(f"Published event {event.event_type} with ID {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    def consume_events(
        self,
        event_handler: Callable[[BaseEvent], None],
        auto_ack: bool = False,
    ) -> None:
        """
        Consume events from RabbitMQ.
        
        Args:
            event_handler: Function to handle received events
            auto_ack: Whether to automatically acknowledge messages
        """
        try:
            if not self.channel or self.channel.is_closed:
                self.connect()
            
            def callback(ch, method, properties, body):
                """Callback function for message processing."""
                try:
                    # Parse message
                    event_data = json.loads(body.decode('utf-8'))
                    event_type = event_data.get("event_type")
                    
                    # Get event class and create instance
                    event_class = get_event_class(EventType(event_type))
                    event = event_class(**event_data)
                    
                    # Handle event
                    event_handler(event)
                    
                    # Acknowledge message
                    if not auto_ack:
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                    
                    logger.info(f"Processed event {event_type} with ID {event.event_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    if not auto_ack:
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            # Set up consumer
            self.channel.basic_qos(prefetch_count=1)
            self.channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=callback,
                auto_ack=auto_ack,
            )
            
            logger.info("Started consuming events from RabbitMQ")
            
            # Start consuming
            self.channel.start_consuming()
            
        except Exception as e:
            logger.error(f"Error consuming events: {e}")
            raise
    
    def stop_consuming(self) -> None:
        """Stop consuming events."""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.stop_consuming()
            logger.info("Stopped consuming events")
        except Exception as e:
            logger.error(f"Error stopping consumption: {e}")


class EventPublisher:
    """Event publisher for API Gateway."""
    
    def __init__(self, rabbitmq_manager: RabbitMQManager):
        """
        Initialize event publisher.
        
        Args:
            rabbitmq_manager: RabbitMQ manager instance
        """
        self.rabbitmq_manager = rabbitmq_manager
    
    def publish_instagram_message_received(
        self,
        tenant_id: UUID,
        instagram_user_id: str,
        instagram_username: str,
        message_id: str,
        message_text: Optional[str] = None,
        media_urls: Optional[list] = None,
        correlation_id: Optional[UUID] = None,
    ) -> bool:
        """
        Publish Instagram direct message received event.
        
        Args:
            tenant_id: The tenant ID
            instagram_user_id: Instagram user ID
            instagram_username: Instagram username
            message_id: Message ID
            message_text: Message text
            media_urls: Media URLs
            correlation_id: Correlation ID
            
        Returns:
            True if published successfully, False otherwise
        """
        try:
            from shared_lib.app.schemas.events import InstagramDirectMessageReceived
            
            event = InstagramDirectMessageReceived(
                tenant_id=tenant_id,
                instagram_user_id=instagram_user_id,
                instagram_username=instagram_username,
                message_id=message_id,
                message_text=message_text,
                media_urls=media_urls or [],
                correlation_id=correlation_id,
                source_service="api_gateway",
            )
            
            return self.rabbitmq_manager.publish_event(event)
            
        except Exception as e:
            logger.error(f"Failed to publish Instagram message event: {e}")
            return False


class EventConsumer:
    """Event consumer for Intelligence Worker."""
    
    def __init__(self, rabbitmq_manager: RabbitMQManager):
        """
        Initialize event consumer.
        
        Args:
            rabbitmq_manager: RabbitMQ manager instance
        """
        self.rabbitmq_manager = rabbitmq_manager
    
    def start_consuming(self, event_handler: Callable[[BaseEvent], None]) -> None:
        """
        Start consuming events.
        
        Args:
            event_handler: Function to handle received events
        """
        try:
            self.rabbitmq_manager.consume_events(event_handler)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping consumer")
            self.rabbitmq_manager.stop_consuming()
        except Exception as e:
            logger.error(f"Error in event consumption: {e}")
            raise


# Global RabbitMQ manager instance
_rabbitmq_manager: Optional[RabbitMQManager] = None
_event_publisher: Optional[EventPublisher] = None
_event_consumer: Optional[EventConsumer] = None


def initialize_rabbitmq(
    host: str,
    port: int,
    username: str,
    password: str,
    vhost: str = "/",
) -> RabbitMQManager:
    """
    Initialize global RabbitMQ manager.
    
    Args:
        host: RabbitMQ host
        port: RabbitMQ port
        username: RabbitMQ username
        password: RabbitMQ password
        vhost: RabbitMQ virtual host
        
    Returns:
        The RabbitMQ manager instance
    """
    global _rabbitmq_manager, _event_publisher, _event_consumer
    
    _rabbitmq_manager = RabbitMQManager(host, port, username, password, vhost)
    _rabbitmq_manager.connect()
    
    _event_publisher = EventPublisher(_rabbitmq_manager)
    _event_consumer = EventConsumer(_rabbitmq_manager)
    
    logger.info("RabbitMQ integration initialized")
    return _rabbitmq_manager


def get_rabbitmq_manager() -> RabbitMQManager:
    """
    Get the global RabbitMQ manager.
    
    Returns:
        The RabbitMQ manager instance
        
    Raises:
        RuntimeError: If RabbitMQ manager is not initialized
    """
    if _rabbitmq_manager is None:
        raise RuntimeError("RabbitMQ manager not initialized")
    return _rabbitmq_manager


def get_event_publisher() -> EventPublisher:
    """
    Get the global event publisher.
    
    Returns:
        The event publisher instance
        
    Raises:
        RuntimeError: If event publisher is not initialized
    """
    if _event_publisher is None:
        raise RuntimeError("Event publisher not initialized")
    return _event_publisher


def get_event_consumer() -> EventConsumer:
    """
    Get the global event consumer.
    
    Returns:
        The event consumer instance
        
    Raises:
        RuntimeError: If event consumer is not initialized
    """
    if _event_consumer is None:
        raise RuntimeError("Event consumer not initialized")
    return _event_consumer
