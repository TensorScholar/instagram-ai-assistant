"""
Aura Platform - Database Models
SQLAlchemy 2.0 models with async support for multi-tenant architecture.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship, DeclarativeBase
from sqlalchemy.sql import func as sql_func


from .mixins import TenantMixin, TimestampMixin

# Base class for all models with async support
class Base(AsyncAttrs, DeclarativeBase):
    __abstract__ = True


class Tenant(Base, TimestampMixin):
    """Tenant model for multi-tenancy."""
    
    __tablename__ = "tenants"
    
    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, server_default='true')
    
    # Instagram configuration
    instagram_business_account_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    instagram_access_token: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    
    # E-commerce configuration
    ecommerce_platform: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )
    secrets_path: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True, comment="Vault path for tenant-specific secrets"
    )
    
    # AI configuration
    ai_model_preference: Mapped[str] = mapped_column(
        String(100), default="gemini-1.5-pro", nullable=False
    )
    brand_voice_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    
    # Relationships
    products: Mapped[List["Product"]] = relationship(
        "Product", back_populates="tenant", cascade="all, delete-orphan"
    )
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="tenant", cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_tenants_domain", "domain"),
        Index("idx_tenants_active", "is_active"),
        Index("idx_tenants_instagram_account", "instagram_business_account_id"),
        {"schema": "shared"},
    )


class Product(Base, TimestampMixin, TenantMixin):
    """Product model for e-commerce products."""
    
    __tablename__ = "products"
    
    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    
    # External system IDs
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    external_platform: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Product details
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    price: Mapped[Optional[float]] = mapped_column(nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    
    # Product metadata
    category: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    images: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    
    # Availability
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    stock_quantity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Vector embedding for semantic search
    embedding_vector: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="products")
    
    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("tenant_id", "external_id", "external_platform"),
        Index("idx_products_tenant_external", "tenant_id", "external_id"),
        Index("idx_products_title", "title"),
        Index("idx_products_category", "category"),
        Index("idx_products_available", "is_available"),
        Index("idx_products_price", "price"),
        {"schema": "tenants"},
    )


class InstagramUser(Base, TimestampMixin, TenantMixin):
    """Instagram user model for tracking user interactions."""
    
    __tablename__ = "instagram_users"
    
    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    
    # Instagram user details
    instagram_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    profile_picture_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # User metadata
    is_business_user: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_interaction: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_messages: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="instagram_user", cascade="all, delete-orphan"
    )
    
    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("tenant_id", "instagram_user_id"),
        Index("idx_instagram_users_tenant_user", "tenant_id", "instagram_user_id"),
        Index("idx_instagram_users_username", "username"),
        Index("idx_instagram_users_last_interaction", "last_interaction"),
        {"schema": "tenants"},
    )


class Conversation(Base, TimestampMixin, TenantMixin):
    """Conversation model for tracking Instagram DM conversations."""
    
    __tablename__ = "conversations"
    
    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    
    # Conversation details
    instagram_user_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("tenants.instagram_users.id", ondelete="CASCADE"),
        nullable=False
    )
    instagram_conversation_id: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    
    # Conversation metadata
    status: Mapped[str] = mapped_column(
        String(50), default="active", nullable=False
    )  # active, closed, archived
    last_message_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="conversations")
    instagram_user: Mapped["InstagramUser"] = relationship(
        "InstagramUser", back_populates="conversations"
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )
    
    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("tenant_id", "instagram_conversation_id"),
        Index("idx_conversations_tenant_user", "tenant_id", "instagram_user_id"),
        Index("idx_conversations_status", "status"),
        Index("idx_conversations_last_message", "last_message_at"),
        {"schema": "tenants"},
    )


class Message(Base, TimestampMixin, TenantMixin):
    """Message model for storing Instagram DM messages."""
    
    __tablename__ = "messages"
    
    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    
    # Message details
    conversation_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("tenants.conversations.id", ondelete="CASCADE"),
        nullable=False
    )
    instagram_message_id: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    
    # Message content
    message_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    message_type: Mapped[str] = mapped_column(
        String(50), default="text", nullable=False
    )  # text, image, video, etc.
    media_urls: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    
    # Message metadata
    sender_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # user, business, system
    is_from_business: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # AI processing
    ai_processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ai_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_model_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )
    
    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("tenant_id", "instagram_message_id"),
        Index("idx_messages_conversation", "conversation_id"),
        Index("idx_messages_sender_type", "sender_type"),
        Index("idx_messages_ai_processed", "ai_processed"),
        Index("idx_messages_created_at", "created_at"),
        {"schema": "tenants"},
    )


class EventLog(Base, TimestampMixin, TenantMixin):
    """Event log model for tracking system events."""
    
    __tablename__ = "event_logs"
    
    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    
    # Event details
    event_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    correlation_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), nullable=False
    )
    
    # Event data
    event_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_service: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Processing status
    processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_event_logs_event_id", "event_id"),
        Index("idx_event_logs_correlation_id", "correlation_id"),
        Index("idx_event_logs_event_type", "event_type"),
        Index("idx_event_logs_processed", "processed"),
        Index("idx_event_logs_source_service", "source_service"),
        {"schema": "shared"},
    )


# Model registry for dynamic model access
MODEL_REGISTRY: Dict[str, type] = {
    "Tenant": Tenant,
    "Product": Product,
    "InstagramUser": InstagramUser,
    "Conversation": Conversation,
    "Message": Message,
    "EventLog": EventLog,
}


def get_model_class(model_name: str) -> type:
    """
    Get the model class for a given model name.
    
    Args:
        model_name: The model name to get the class for
        
    Returns:
        The model class
        
    Raises:
        ValueError: If the model name is not registered
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not registered")
    return MODEL_REGISTRY[model_name]
