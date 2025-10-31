"""
Aura Platform - API Gateway Configuration
Configuration management for the API Gateway service.
"""

import logging
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """API Gateway configuration settings."""
    
    # Application settings
    app_name: str = Field(default="Aura API Gateway")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    reload: bool = Field(default=True)
    
    # Database settings
    database_url: Optional[str] = Field(default=None)
    postgres_host: str = Field(default="postgres")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="aura_platform")
    postgres_user: str = Field(default="aura_user")
    postgres_password: Optional[str] = Field(default=None)
    
    # Message queue settings
    rabbitmq_host: str = Field(default="rabbitmq")
    rabbitmq_port: int = Field(default=5672)
    rabbitmq_user: str = Field(default="aura_user")
    rabbitmq_password: Optional[str] = Field(default=None)
    rabbitmq_vhost: str = Field(default="aura_vhost")
    # RabbitMQ management API (for back-pressure)
    rabbitmq_mgmt_url: str = Field(default="http://rabbitmq:15672")
    # For security, do not default to guest credentials; require explicit configuration
    rabbitmq_mgmt_user: Optional[str] = Field(default=None)
    rabbitmq_mgmt_password: Optional[str] = Field(default=None)
    
    # Security settings
    secret_key: Optional[str] = Field(default=None)
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(
        default=30
    )
    
    # Instagram API settings
    instagram_app_id: Optional[str] = Field(default=None)
    instagram_app_secret: Optional[str] = Field(default=None)
    instagram_webhook_verify_token: Optional[str] = Field(default=None)
    instagram_graph_api_version: str = Field(default="v18.0")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"]
    )
    cors_allow_credentials: bool = Field(default=True)
    
    # Redis settings (optional for distributed rate limiting)
    redis_host: str = Field(default="redis")
    redis_port: int = Field(default=6379)
    redis_password: Optional[str] = Field(default=None)
    redis_db: int = Field(default=0)

    # Proxy/middleware security
    trusted_proxy_subnets: List[str] = Field(default_factory=list)
    
    # Logging settings
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


# Global settings instance
settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info(f"API Gateway configuration loaded for {settings.environment} environment")
