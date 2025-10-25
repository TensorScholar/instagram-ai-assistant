"""
Aura Platform - Event Schemas
Pydantic schemas for event-driven communication between services.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


class EventType(str, Enum):
    """Event types for the Aura platform."""
    
    # Instagram Events
    INSTAGRAM_DIRECT_MESSAGE_RECEIVED = "instagram.direct_message.received"
    INSTAGRAM_DIRECT_MESSAGE_SENT = "instagram.direct_message.sent"
    INSTAGRAM_MEDIA_UPLOADED = "instagram.media.uploaded"
    
    # Product Events
    PRODUCT_SYNC_STARTED = "product.sync.started"
    PRODUCT_SYNC_COMPLETED = "product.sync.completed"
    PRODUCT_SYNC_FAILED = "product.sync.failed"
    PRODUCT_UPDATED = "product.updated"
    PRODUCT_DELETED = "product.deleted"
    
    # AI Processing Events
    AI_QUERY_RECEIVED = "ai.query.received"
    AI_QUERY_PROCESSED = "ai.query.processed"
    AI_QUERY_FAILED = "ai.query.failed"
    
    # Tenant Events
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"
    TENANT_DEACTIVATED = "tenant.deactivated"


class BaseEvent(BaseModel):
    """Base event schema with common fields."""
    
    event_id: UUID = Field(default_factory=lambda: UUID.uuid4())
    event_type: EventType
    tenant_id: UUID
    correlation_id: UUID = Field(default_factory=lambda: UUID.uuid4())
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str
    version: str = Field(default="1.0")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class InstagramDirectMessageReceived(BaseEvent):
    """Event schema for Instagram direct message received."""
    
    event_type: EventType = Field(default=EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED)
    
    # Instagram specific fields
    instagram_user_id: str
    instagram_username: str
    message_id: str
    message_text: Optional[str] = None
    message_type: str = Field(default="text")  # text, image, video, etc.
    media_urls: List[str] = Field(default_factory=list)
    
    # Message metadata
    is_business_message: bool = Field(default=True)
    reply_to_message_id: Optional[str] = None
    
    @validator('message_text')
    def validate_message_content(cls, v, values):
        """Validate that either text or media is present."""
        if not v and not values.get('media_urls'):
            raise ValueError('Either message_text or media_urls must be provided')
        return v


class InstagramDirectMessageSent(BaseEvent):
    """Event schema for Instagram direct message sent."""
    
    event_type: EventType = Field(default=EventType.INSTAGRAM_DIRECT_MESSAGE_SENT)
    
    # Instagram specific fields
    instagram_user_id: str
    message_id: str
    message_text: str
    message_type: str = Field(default="text")
    
    # Response metadata
    response_time_ms: Optional[int] = None
    ai_model_used: Optional[str] = None


class ProductSyncStarted(BaseEvent):
    """Event schema for product synchronization started."""
    
    event_type: EventType = Field(default=EventType.PRODUCT_SYNC_STARTED)
    
    # Sync metadata
    sync_id: UUID = Field(default_factory=lambda: UUID.uuid4())
    connector_type: str  # shopify, woocommerce, etc.
    total_products: int
    batch_size: int = Field(default=100)


class ProductSyncCompleted(BaseEvent):
    """Event schema for product synchronization completed."""
    
    event_type: EventType = Field(default=EventType.PRODUCT_SYNC_COMPLETED)
    
    # Sync results
    sync_id: UUID
    products_processed: int
    products_created: int
    products_updated: int
    products_failed: int
    duration_seconds: float


class ProductSyncFailed(BaseEvent):
    """Event schema for product synchronization failed."""
    
    event_type: EventType = Field(default=EventType.PRODUCT_SYNC_FAILED)
    
    # Error details
    sync_id: UUID
    error_message: str
    error_code: Optional[str] = None
    retry_count: int = Field(default=0)


class AIQueryReceived(BaseEvent):
    """Event schema for AI query received."""
    
    event_type: EventType = Field(default=EventType.AI_QUERY_RECEIVED)
    
    # Query details
    query_id: UUID = Field(default_factory=lambda: UUID.uuid4())
    query_text: str
    query_type: str = Field(default="text")  # text, image, hybrid
    media_urls: List[str] = Field(default_factory=list)
    
    # Context
    instagram_user_id: str
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)


class AIQueryProcessed(BaseEvent):
    """Event schema for AI query processed."""
    
    event_type: EventType = Field(default=EventType.AI_QUERY_PROCESSED)
    
    # Response details
    query_id: UUID
    response_text: str
    response_type: str = Field(default="text")
    product_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Processing metadata
    processing_time_ms: int
    ai_model_used: str
    confidence_score: Optional[float] = None


class AIQueryFailed(BaseEvent):
    """Event schema for AI query failed."""
    
    event_type: EventType = Field(default=EventType.AI_QUERY_FAILED)
    
    # Error details
    query_id: UUID
    error_message: str
    error_code: Optional[str] = None
    retry_count: int = Field(default=0)


class TenantCreated(BaseEvent):
    """Event schema for tenant created."""
    
    event_type: EventType = Field(default=EventType.TENANT_CREATED)
    
    # Tenant details
    tenant_name: str
    tenant_domain: str
    instagram_business_account_id: str
    ecommerce_platform: str
    ecommerce_credentials: Dict[str, Any]


class TenantUpdated(BaseEvent):
    """Event schema for tenant updated."""
    
    event_type: EventType = Field(default=EventType.TENANT_UPDATED)
    
    # Update details
    updated_fields: List[str]
    old_values: Dict[str, Any] = Field(default_factory=dict)
    new_values: Dict[str, Any] = Field(default_factory=dict)


class TenantDeactivated(BaseEvent):
    """Event schema for tenant deactivated."""
    
    event_type: EventType = Field(default=EventType.TENANT_DEACTIVATED)
    
    # Deactivation details
    deactivation_reason: str
    deactivation_date: datetime = Field(default_factory=datetime.utcnow)


# Event registry for dynamic event handling
EVENT_REGISTRY: Dict[EventType, type] = {
    EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED: InstagramDirectMessageReceived,
    EventType.INSTAGRAM_DIRECT_MESSAGE_SENT: InstagramDirectMessageSent,
    EventType.PRODUCT_SYNC_STARTED: ProductSyncStarted,
    EventType.PRODUCT_SYNC_COMPLETED: ProductSyncCompleted,
    EventType.PRODUCT_SYNC_FAILED: ProductSyncFailed,
    EventType.AI_QUERY_RECEIVED: AIQueryReceived,
    EventType.AI_QUERY_PROCESSED: AIQueryProcessed,
    EventType.AI_QUERY_FAILED: AIQueryFailed,
    EventType.TENANT_CREATED: TenantCreated,
    EventType.TENANT_UPDATED: TenantUpdated,
    EventType.TENANT_DEACTIVATED: TenantDeactivated,
}


def get_event_class(event_type: EventType) -> type:
    """
    Get the event class for a given event type.
    
    Args:
        event_type: The event type to get the class for
        
    Returns:
        The event class
        
    Raises:
        ValueError: If the event type is not registered
    """
    if event_type not in EVENT_REGISTRY:
        raise ValueError(f"Event type {event_type} is not registered")
    return EVENT_REGISTRY[event_type]


def create_event(event_type: EventType, **kwargs) -> BaseEvent:
    """
    Create an event instance of the specified type.
    
    Args:
        event_type: The type of event to create
        **kwargs: Additional arguments for the event
        
    Returns:
        The created event instance
    """
    event_class = get_event_class(event_type)
    return event_class(**kwargs)