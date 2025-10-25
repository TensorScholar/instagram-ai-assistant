"""
Aura Platform - API Gateway Main Application
FastAPI application for the API Gateway service.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.endpoints import webhooks
from .core.config import settings
from shared_lib.app.utils.security import initialize_security

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Args:
        app: The FastAPI application instance
        
    Yields:
        None
    """
    # Startup
    logger.info("Starting API Gateway service")
    
    # TODO: Initialize database connection
    # TODO: Initialize RabbitMQ connection
    initialize_security(
        settings.secret_key,
        settings.instagram_app_secret,
        None
    )
    
    logger.info("API Gateway service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway service")
    
    # TODO: Close database connection
    # TODO: Close RabbitMQ connection
    
    logger.info("API Gateway service shut down")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API Gateway for Aura Platform - Instagram AI Assistant",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(webhooks.router)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler.
    
    Args:
        request: The FastAPI request object
        exc: The exception that occurred
        
    Returns:
        JSON error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": "unknown",  # TODO: Add request ID tracking
        }
    )


@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        Welcome message
    """
    return {
        "message": "Welcome to Aura Platform API Gateway",
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
    }


@app.get("/info")
async def info():
    """
    Service information endpoint.
    
    Returns:
        Service information
    """
    return {
        "service": "api_gateway",
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "host": settings.host,
        "port": settings.port,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
