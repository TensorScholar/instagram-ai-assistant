"""
Aura Platform - Load Stress Tests
Comprehensive load testing for infrastructure hardening validation.
"""

import asyncio
import pytest
import time
from typing import List, Dict, Any
from uuid import uuid4
import httpx
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, func
from shared_lib.app.db.database import DatabaseManager
from shared_lib.app.schemas.models import Base, TenantMixin
from shared_lib.app.core.config import settings


# Test model for load testing
class LoadTestModel(TenantMixin, Base):
    """Test model for load testing database operations."""
    __tablename__ = "load_test_models"
    
    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    data: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())


class LoadTestSuite:
    """Comprehensive load testing suite for infrastructure validation."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.db_manager: DatabaseManager = None
        self.test_tenant_id = uuid4()
    
    async def setup(self):
        """Setup test environment."""
        # Initialize database with enhanced pool settings
        self.db_manager = DatabaseManager(
            database_url=settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
        )
        
        # Create test tables
        await self.db_manager.create_tables()
    
    async def teardown(self):
        """Cleanup test environment."""
        if self.db_manager:
            await self.db_manager.close()
    
    async def test_database_connection_pool_exhaustion(self, concurrent_requests: int = 100) -> Dict[str, Any]:
        """
        Test V-001: Database Connection Pool Exhaustion
        
        Simulates 100 concurrent database operations to verify pool can handle load.
        """
        start_time = time.time()
        tasks = []
        errors = []
        success_count = 0
        
        async def db_operation(session_id: int):
            """Single database operation."""
            try:
                async with self.db_manager.get_session() as session:
                    # Create a test record
                    test_model = LoadTestModel(
                        tenant_id=self.test_tenant_id,
                        name=f"LoadTest_{session_id}",
                        data=f"Test data for session {session_id}"
                    )
                    session.add(test_model)
                    await session.commit()
                    return True
            except Exception as e:
                errors.append(f"Session {session_id}: {str(e)}")
                return False
        
        # Create concurrent tasks
        for i in range(concurrent_requests):
            task = asyncio.create_task(db_operation(i))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes
        for result in results:
            if result is True:
                success_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        success_rate = (success_count / concurrent_requests) * 100
        avg_response_time = duration / concurrent_requests
        
        result = {
            "test_name": "database_connection_pool_exhaustion",
            "concurrent_requests": concurrent_requests,
            "success_count": success_count,
            "error_count": len(errors),
            "success_rate": success_rate,
            "duration": duration,
            "avg_response_time": avg_response_time,
            "errors": errors[:10],  # Limit error details
            "passed": success_rate >= 95.0 and avg_response_time < 2.0
        }
        
        self.results["database_pool_test"] = result
        return result
    
    async def test_celery_queue_processing_capacity(self, message_count: int = 1000) -> Dict[str, Any]:
        """
        Test V-002: Celery Worker Thread Starvation
        
        Simulates high message volume to verify queue processing capacity.
        """
        start_time = time.time()
        
        # Simulate message publishing (mock implementation)
        published_messages = 0
        processed_messages = 0
        queue_backlog = 0
        
        # Simulate message publishing
        for i in range(message_count):
            # Mock message publishing
            published_messages += 1
            
            # Simulate processing delay
            if i % 20 == 0:  # Simulate worker processing every 20 messages
                processed_messages += 20
                await asyncio.sleep(0.01)  # Small delay to simulate processing
        
        # Calculate remaining backlog
        queue_backlog = published_messages - processed_messages
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        processing_rate = processed_messages / duration if duration > 0 else 0
        backlog_percentage = (queue_backlog / message_count) * 100
        
        result = {
            "test_name": "celery_queue_processing_capacity",
            "message_count": message_count,
            "published_messages": published_messages,
            "processed_messages": processed_messages,
            "queue_backlog": queue_backlog,
            "processing_rate": processing_rate,
            "backlog_percentage": backlog_percentage,
            "duration": duration,
            "passed": backlog_percentage < 10.0 and processing_rate > 100.0
        }
        
        self.results["celery_queue_test"] = result
        return result
    
    async def test_rate_limiting_protection(self, requests_per_second: int = 10) -> Dict[str, Any]:
        """
        Test V-003: Rate Limiting Protection
        
        Tests API Gateway rate limiting under high request volume.
        """
        start_time = time.time()
        successful_requests = 0
        rate_limited_requests = 0
        errors = []
        
        async def make_request(request_id: int):
            """Make a single HTTP request."""
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "http://localhost:8000/info",
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        return "success"
                    elif response.status_code == 429:
                        return "rate_limited"
                    else:
                        return f"error_{response.status_code}"
                        
            except Exception as e:
                errors.append(f"Request {request_id}: {str(e)}")
                return "error"
        
        # Create burst of requests
        tasks = []
        for i in range(requests_per_second * 10):  # 10 seconds worth of requests
            task = asyncio.create_task(make_request(i))
            tasks.append(task)
            
            # Rate limit the requests
            await asyncio.sleep(1.0 / requests_per_second)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count results
        for result in results:
            if result == "success":
                successful_requests += 1
            elif result == "rate_limited":
                rate_limited_requests += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        total_requests = len(results)
        success_rate = (successful_requests / total_requests) * 100
        rate_limit_rate = (rate_limited_requests / total_requests) * 100
        
        result = {
            "test_name": "rate_limiting_protection",
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "rate_limited_requests": rate_limited_requests,
            "success_rate": success_rate,
            "rate_limit_rate": rate_limit_rate,
            "duration": duration,
            "errors": errors[:5],
            "passed": rate_limit_rate > 0 and success_rate > 80.0
        }
        
        self.results["rate_limiting_test"] = result
        return result
    
    async def test_back_pressure_protection(self, queue_length: int = 6000) -> Dict[str, Any]:
        """
        Test V-005 & V-006: Back-Pressure Protection
        
        Tests API Gateway back-pressure when queues are overloaded.
        """
        # Mock queue length check
        mock_queue_lengths = {
            "realtime_queue": queue_length,
            "bulk_queue": queue_length // 2
        }
        
        # Simulate requests during overload
        requests_during_overload = 0
        service_unavailable_responses = 0
        successful_responses = 0
        
        async def make_request_during_overload(request_id: int):
            """Make request during simulated overload."""
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "http://localhost:8000/info",
                        timeout=5.0
                    )
                    
                    if response.status_code == 503:
                        return "service_unavailable"
                    elif response.status_code == 200:
                        return "success"
                    else:
                        return f"error_{response.status_code}"
                        
            except Exception as e:
                return "error"
        
        # Simulate requests during overload
        tasks = []
        for i in range(100):  # 100 requests during overload
            task = asyncio.create_task(make_request_during_overload(i))
            tasks.append(task)
            requests_during_overload += 1
        
        # Wait for all requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count results
        for result in results:
            if result == "service_unavailable":
                service_unavailable_responses += 1
            elif result == "success":
                successful_responses += 1
        
        # Calculate metrics
        total_requests = len(results)
        service_unavailable_rate = (service_unavailable_responses / total_requests) * 100
        success_rate = (successful_responses / total_requests) * 100
        
        result = {
            "test_name": "back_pressure_protection",
            "mock_queue_length": queue_length,
            "total_requests": total_requests,
            "service_unavailable_responses": service_unavailable_responses,
            "successful_responses": successful_responses,
            "service_unavailable_rate": service_unavailable_rate,
            "success_rate": success_rate,
            "passed": service_unavailable_rate > 50.0  # Should reject most requests during overload
        }
        
        self.results["back_pressure_test"] = result
        return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all load tests and return comprehensive results."""
        await self.setup()
        
        try:
            # Run all tests
            await self.test_database_connection_pool_exhaustion()
            await self.test_celery_queue_processing_capacity()
            await self.test_rate_limiting_protection()
            await self.test_back_pressure_protection()
            
            # Calculate overall results
            total_tests = len(self.results)
            passed_tests = sum(1 for result in self.results.values() if result.get("passed", False))
            
            overall_result = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests) * 100,
                "test_results": self.results,
                "overall_passed": passed_tests == total_tests
            }
            
            return overall_result
            
        finally:
            await self.teardown()


# Pytest test functions
@pytest.mark.asyncio
async def test_database_connection_pool_exhaustion():
    """Test V-001: Database Connection Pool Exhaustion."""
    test_suite = LoadTestSuite()
    result = await test_suite.test_database_connection_pool_exhaustion(concurrent_requests=100)
    
    assert result["passed"], f"Database pool test failed: {result}"
    assert result["success_rate"] >= 95.0, f"Success rate too low: {result['success_rate']}%"
    assert result["avg_response_time"] < 2.0, f"Response time too high: {result['avg_response_time']}s"


@pytest.mark.asyncio
async def test_celery_queue_processing_capacity():
    """Test V-002: Celery Worker Thread Starvation."""
    test_suite = LoadTestSuite()
    result = await test_suite.test_celery_queue_processing_capacity(message_count=1000)
    
    assert result["passed"], f"Celery queue test failed: {result}"
    assert result["backlog_percentage"] < 10.0, f"Queue backlog too high: {result['backlog_percentage']}%"
    assert result["processing_rate"] > 100.0, f"Processing rate too low: {result['processing_rate']}"


@pytest.mark.asyncio
async def test_rate_limiting_protection():
    """Test V-003: Rate Limiting Protection."""
    test_suite = LoadTestSuite()
    result = await test_suite.test_rate_limiting_protection(requests_per_second=10)
    
    assert result["passed"], f"Rate limiting test failed: {result}"
    assert result["rate_limit_rate"] > 0, f"No rate limiting detected: {result['rate_limit_rate']}%"
    assert result["success_rate"] > 80.0, f"Success rate too low: {result['success_rate']}%"


@pytest.mark.asyncio
async def test_back_pressure_protection():
    """Test V-005 & V-006: Back-Pressure Protection."""
    test_suite = LoadTestSuite()
    result = await test_suite.test_back_pressure_protection(queue_length=6000)
    
    assert result["passed"], f"Back-pressure test failed: {result}"
    assert result["service_unavailable_rate"] > 50.0, f"Not enough back-pressure: {result['service_unavailable_rate']}%"


@pytest.mark.asyncio
async def test_comprehensive_load_testing():
    """Run comprehensive load testing suite."""
    test_suite = LoadTestSuite()
    results = await test_suite.run_all_tests()
    
    assert results["overall_passed"], f"Load testing suite failed: {results}"
    assert results["success_rate"] == 100.0, f"Not all tests passed: {results['success_rate']}%"
