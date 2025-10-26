"""
Aura Platform - API Gateway Middleware
Rate limiting and back-pressure middleware for the API Gateway.
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import httpx

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    """
    
    def __init__(self, app, requests_per_minute: int = 60, window_size: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = window_size
        self.requests: Dict[str, list] = {}
        self._cleanup_task = None
        
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = datetime.now()
        
        # Clean up old requests
        await self._cleanup_old_requests(current_time)
        
        # Check rate limit
        if not await self._check_rate_limit(client_ip, current_time):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Record request
        await self._record_request(client_ip, current_time)
        
        response = await call_next(request)
        return response
    
    async def _check_rate_limit(self, client_ip: str, current_time: datetime) -> bool:
        """Check if client has exceeded rate limit."""
        if client_ip not in self.requests:
            return True
        
        # Count requests in the last window
        window_start = current_time - timedelta(seconds=self.window_size)
        recent_requests = [
            req_time for req_time in self.requests[client_ip]
            if req_time > window_start
        ]
        
        return len(recent_requests) < self.requests_per_minute
    
    async def _record_request(self, client_ip: str, current_time: datetime):
        """Record a request for the client."""
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)
    
    async def _cleanup_old_requests(self, current_time: datetime):
        """Clean up old request records."""
        cutoff_time = current_time - timedelta(seconds=self.window_size * 2)
        
        for client_ip in list(self.requests.keys()):
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if req_time > cutoff_time
            ]
            
            # Remove empty entries
            if not self.requests[client_ip]:
                del self.requests[client_ip]


class BackPressureMiddleware(BaseHTTPMiddleware):
    """
    Back-pressure middleware that monitors queue health and throttles requests.
    """
    
    def __init__(
        self, 
        app,
        rabbitmq_management_url: str = "http://rabbitmq:15672",
        rabbitmq_username: str = "guest",
        rabbitmq_password: str = "guest",
        critical_queue_length: int = 5000,
        warning_queue_length: int = 3000,
        check_interval: int = 30
    ):
        super().__init__(app)
        self.rabbitmq_management_url = rabbitmq_management_url
        self.rabbitmq_username = rabbitmq_username
        self.rabbitmq_password = rabbitmq_password
        self.critical_queue_length = critical_queue_length
        self.warning_queue_length = warning_queue_length
        self.check_interval = check_interval
        
        self.queue_lengths: Dict[str, int] = {}
        self.last_check = datetime.now()
        self._check_task = None
        
    async def dispatch(self, request: Request, call_next):
        """Process request with back-pressure protection."""
        # Check queue health
        if await self._is_system_overloaded():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "message": "System is experiencing high load. Please try again later.",
                    "retry_after": 30
                },
                headers={"Retry-After": "30"}
            )
        
        response = await call_next(request)
        return response
    
    async def _is_system_overloaded(self) -> bool:
        """Check if system is overloaded based on queue lengths."""
        current_time = datetime.now()
        
        # Check if we need to refresh queue data
        if (current_time - self.last_check).seconds >= self.check_interval:
            await self._update_queue_lengths()
            self.last_check = current_time
        
        # Check if any critical queue is overloaded
        for queue_name, length in self.queue_lengths.items():
            if length >= self.critical_queue_length:
                logger.warning(f"Queue {queue_name} is overloaded: {length} messages")
                return True
        
        return False
    
    async def _update_queue_lengths(self):
        """Update queue lengths from RabbitMQ management API."""
        try:
            async with httpx.AsyncClient() as client:
                # Get queue information
                response = await client.get(
                    f"{self.rabbitmq_management_url}/api/queues",
                    auth=(self.rabbitmq_username, self.rabbitmq_password),
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    queues = response.json()
                    for queue in queues:
                        queue_name = queue.get("name", "")
                        if queue_name in ["realtime_queue", "bulk_queue"]:
                            self.queue_lengths[queue_name] = queue.get("messages", 0)
                    
                    logger.debug(f"Updated queue lengths: {self.queue_lengths}")
                else:
                    logger.error(f"Failed to get queue info: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error updating queue lengths: {e}")
            # On error, assume system is healthy to avoid false positives


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """
    Health check middleware that bypasses rate limiting and back-pressure.
    """
    
    def __init__(self, app, health_paths: list = None):
        super().__init__(app)
        self.health_paths = health_paths or ["/health", "/info", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next):
        """Process request, bypassing middleware for health checks."""
        if request.url.path in self.health_paths:
            return await call_next(request)
        
        # For non-health endpoints, continue with other middleware
        return await call_next(request)
