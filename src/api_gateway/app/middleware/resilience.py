"""
Aura Platform - API Gateway Middleware
Rate limiting and back-pressure middleware for the API Gateway.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import ipaddress
try:
    import redis.asyncio as redis
except Exception:
    redis = None  # Optional dependency
from ..core.config import settings as api_settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        window_size: int = 60,
        use_redis: bool = False,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        redis_db: int = 0,
        key_prefix: str = "ratelimit",
        trusted_proxy_headers: Optional[List[str]] = None,
        trusted_proxy_subnets: Optional[List[str]] = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = window_size
        self.requests: Dict[str, list] = {}
        self._max_keys = 10000  # bound memory
        self._cleanup_task = None
        self.key_prefix = key_prefix
        self.trusted_proxy_headers = trusted_proxy_headers or ["X-Forwarded-For"]
        # Subnets that are allowed to set X-Forwarded-For (e.g., ingress/LB ranges)
        self.trusted_proxy_subnets = [ipaddress.ip_network(s) for s in (trusted_proxy_subnets or [])]
        self._redis = None
        if use_redis and redis is not None and redis_host:
            try:
                self._redis = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    db=redis_db,
                    decode_responses=True,
                )
            except Exception:
                self._redis = None
        
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        client_ip = self._get_client_ip(request)
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
        # Redis fixed window per-minute key
        if self._redis:
            try:
                window = current_time.replace(second=0, microsecond=0).isoformat()
                key = f"{self.key_prefix}:{client_ip}:{window}"
                count = await self._redis.incr(key)
                if count == 1:
                    await self._redis.expire(key, self.window_size)
                return count <= self.requests_per_minute
            except Exception:
                # Redis not usable; permanently fall back to in-memory for this process
                self._redis = None
        
        if client_ip not in self.requests:
            return True
        window_start = current_time - timedelta(seconds=self.window_size)
        recent_requests = [t for t in self.requests[client_ip] if t > window_start]
        return len(recent_requests) < self.requests_per_minute
    
    async def _record_request(self, client_ip: str, current_time: datetime):
        """Record a request for the client."""
        if self._redis:
            return  # already counted
        if client_ip not in self.requests:
            # bound keys to avoid unbounded growth
            if len(self.requests) >= self._max_keys:
                # remove oldest key
                oldest = next(iter(self.requests))
                self.requests.pop(oldest, None)
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)
    
    async def _cleanup_old_requests(self, current_time: datetime):
        """Clean up old request records."""
        cutoff_time = current_time - timedelta(seconds=self.window_size * 2)
        for client_ip in list(self.requests.keys()):
            self.requests[client_ip] = [t for t in self.requests[client_ip] if t > cutoff_time]
            if not self.requests[client_ip]:
                self.requests.pop(client_ip, None)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP safely.

        If the immediate peer is within a trusted proxy subnet, honor
        the first X-Forwarded-For entry when present and valid.
        """
        peer_ip = request.client.host if request.client else "unknown"
        try:
            peer_ip_obj = ipaddress.ip_address(peer_ip)
        except Exception:
            peer_ip_obj = None

        peer_is_trusted = False
        if peer_ip_obj and self.trusted_proxy_subnets:
            peer_is_trusted = any(peer_ip_obj in subnet for subnet in self.trusted_proxy_subnets)

        if peer_is_trusted:
            for header in self.trusted_proxy_headers:
                xff = request.headers.get(header)
                if not xff:
                    continue
                first_ip = xff.split(",")[0].strip()
                try:
                    # Ensure it parses as an IP; do not allow private ranges here
                    ip_obj = ipaddress.ip_address(first_ip)
                    return first_ip
                except Exception:
                    continue

        return peer_ip


class BackPressureMiddleware(BaseHTTPMiddleware):
    """
    Back-pressure middleware that monitors queue health and throttles requests.
    """
    
    def __init__(
        self, 
        app,
        rabbitmq_management_url: str = "http://rabbitmq:15672",
        rabbitmq_username: Optional[str] = None,
        rabbitmq_password: Optional[str] = None,
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
        if (current_time - self.last_check).seconds >= self.check_interval or not self.queue_lengths:
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
                kwargs = {"timeout": 5.0}
                if self.rabbitmq_username and self.rabbitmq_password:
                    kwargs["auth"] = (self.rabbitmq_username, self.rabbitmq_password)
                response = await client.get(
                    f"{self.rabbitmq_management_url}/api/queues",
                    **kwargs,
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
        self.health_paths = health_paths or ["/health", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next):
        """Process request, bypassing middleware for health checks."""
        if request.url.path in self.health_paths:
            # Short-circuit minimal responses for /health and /info
            if request.url.path == "/health":
                return JSONResponse({
                    "status": "healthy",
                    "service": "api_gateway",
                    "version": api_settings.app_version,
                    "environment": api_settings.environment,
                })
            if request.url.path == "/info":
                return JSONResponse({
                    "service": "api_gateway",
                    "version": api_settings.app_version,
                    "environment": api_settings.environment,
                    "debug": api_settings.debug,
                    "host": api_settings.host,
                    "port": api_settings.port,
                })
            return await call_next(request)
        
        # For non-health endpoints, continue with other middleware
        return await call_next(request)
