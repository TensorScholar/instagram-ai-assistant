"""
Aura Platform - API Gateway Configuration
Configuration management for the API Gateway service.
"""

import logging
from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """API Gateway configuration settings."""
    
    # Application settings
    app_name: str = Field(default="Aura API Gateway", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="API_GATEWAY_HOST")
    port: int = Field(default=8000, env="API_GATEWAY_PORT")
    workers: int = Field(default=4, env="API_GATEWAY_WORKERS")
    reload: bool = Field(default=True, env="API_GATEWAY_RELOAD")
    
    # Database settings
    database_url: str = Field(env="DATABASE_URL")
    postgres_host: str = Field(default="postgres", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="aura_platform", env="POSTGRES_DB")
    postgres_user: str = Field(default="aura_user", env="POSTGRES_USER")
    postgres_password: str = Field(env="POSTGRES_PASSWORD")
    
    # Message queue settings
    rabbitmq_host: str = Field(default="rabbitmq", env="RABBITMQ_HOST")
    rabbitmq_port: int = Field(default=5672, env="RABBITMQ_PORT")
    rabbitmq_user: str = Field(default="aura_user", env="RABBITMQ_USER")
    rabbitmq_password: str = Field(env="RABBITMQ_PASSWORD")
    rabbitmq_vhost: str = Field(default="aura_vhost", env="RABBITMQ_VHOST")
    
    # Security settings
    secret_key: str = Field(env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(
        default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # Instagram API settings
    instagram_app_id: str = Field(env="INSTAGRAM_APP_ID")
    instagram_app_secret: str = Field(env="INSTAGRAM_APP_SECRET")
    instagram_webhook_verify_token: str = Field(env="INSTAGRAM_WEBHOOK_VERIFY_TOKEN")
    instagram_graph_api_version: str = Field(default="v18.0", env="INSTAGRAM_GRAPH_API_VERSION")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"], env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info(f"API Gateway configuration loaded for {settings.environment} environment")
