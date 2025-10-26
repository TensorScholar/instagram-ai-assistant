"""
Aura Platform - Shared Library Configuration

Centralized configuration management for all shared library components.
Provides type-safe settings with environment variable integration,
database connection pooling, and service-specific configurations.
"""

import logging
from typing import List, Optional

from pydantic import BaseSettings, Field

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Shared library configuration settings."""
    
    # Application settings
    app_name: str = Field(default="Aura Platform", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Database settings
    database_url: str = Field(env="DATABASE_URL")
    postgres_host: str = Field(default="postgres", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="aura_platform", env="POSTGRES_DB")
    postgres_user: str = Field(default="aura_user", env="POSTGRES_USER")
    postgres_password: str = Field(env="POSTGRES_PASSWORD")
    
    # Database connection pool settings
    db_pool_size: int = Field(default=50, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=100, env="DB_MAX_OVERFLOW")
    db_pool_pre_ping: bool = Field(default=True, env="DB_POOL_PRE_PING")
    db_pool_recycle: int = Field(default=1800, env="DB_POOL_RECYCLE")  # 30 minutes
    
    # Redis settings
    redis_host: str = Field(default="redis", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: str = Field(env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # RabbitMQ settings
    rabbitmq_host: str = Field(default="rabbitmq", env="RABBITMQ_HOST")
    rabbitmq_port: int = Field(default=5672, env="RABBITMQ_PORT")
    rabbitmq_username: str = Field(default="aura_user", env="RABBITMQ_USERNAME")
    rabbitmq_password: str = Field(env="RABBITMQ_PASSWORD")
    rabbitmq_vhost: str = Field(default="aura_vhost", env="RABBITMQ_VHOST")
    
    # Vector database settings
    milvus_host: str = Field(default="milvus", env="MILVUS_HOST")
    milvus_port: int = Field(default=19530, env="MILVUS_PORT")
    milvus_user: str = Field(default="root", env="MILVUS_USER")
    milvus_password: str = Field(default="Milvus", env="MILVUS_PASSWORD")
    
    # AI/LLM settings
    gemini_api_key: str = Field(env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    
    # Security settings
    secret_key: str = Field(env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
