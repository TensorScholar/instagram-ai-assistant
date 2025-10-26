"""
Aura Platform - Transaction Integration Tests
Integration tests for transaction rollback and network failure handling.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.exc import OperationalError, DisconnectionError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, func
from uuid import uuid4
from datetime import datetime

from src.shared_lib.app.db.database import transactional, DatabaseTransactionError, DatabaseManager
from src.shared_lib.app.schemas.models import Base, TenantMixin


# Test model for transaction testing
class TransactionTestModel(TenantMixin, Base):
    """Test model for transaction testing."""
    __tablename__ = "transaction_test_models"
    
    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    data: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())


class TransactionTestSuite:
    """Comprehensive transaction testing suite."""
    
    def __init__(self):
        self.db_manager: DatabaseManager = None
        self.test_tenant_id = uuid4()
    
    async def setup(self):
        """Setup test environment."""
        # Initialize database
        self.db_manager = DatabaseManager(
            database_url="sqlite+aiosqlite:///:memory:",
            pool_size=5,
            max_overflow=10,
        )
        
        # Create test tables
        await self.db_manager.create_tables()
    
    async def teardown(self):
        """Cleanup test environment."""
        if self.db_manager:
            await self.db_manager.close()
    
    async def test_transaction_rollback_on_network_failure(self):
        """
        Test V-007: Transaction Rollback Failure
        
        Tests that transactions properly rollback on network failures.
        """
        # Mock database session
        mock_session = AsyncMock()
        mock_session.in_transaction.return_value = False
        
        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_session.begin.return_value = mock_transaction
        mock_transaction.__aenter__ = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock()
        
        # Simulate network failure during commit
        mock_session.commit.side_effect = OperationalError(
            "connection to server at 'localhost' (127.0.0.1), port 5432 failed",
            None,
            None
        )
        
        @transactional
        async def test_function(session: AsyncSession):
            # Simulate multi-table operation
            test_model = TransactionTestModel(
                tenant_id=self.test_tenant_id,
                name="Test Model",
                data="Test data"
            )
            session.add(test_model)
            await session.commit()  # This will fail
            return "success"
        
        # Execute function and expect rollback
        with pytest.raises(DatabaseTransactionError) as exc_info:
            await test_function(mock_session)
        
        # Verify rollback was called
        mock_session.rollback.assert_called_once()
        
        # Verify commit was attempted
        assert mock_session.commit.call_count == 1
        
        # Verify the exception message
        assert "network failure" in str(exc_info.value)
        
        return {
            "test_name": "transaction_rollback_on_network_failure",
            "rollback_called": mock_session.rollback.called,
            "commit_attempted": mock_session.commit.call_count == 1,
            "exception_type": type(exc_info.value).__name__,
            "passed": True
        }
    
    async def test_transaction_rollback_on_disconnection_error(self):
        """
        Test V-007: Transaction Rollback on DisconnectionError
        
        Tests that transactions properly rollback on disconnection errors.
        """
        # Mock database session
        mock_session = AsyncMock()
        mock_session.in_transaction.return_value = False
        
        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_session.begin.return_value = mock_transaction
        mock_transaction.__aenter__ = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock()
        
        # Simulate disconnection error during commit
        mock_session.commit.side_effect = DisconnectionError(
            "connection to server lost"
        )
        
        @transactional
        async def test_function(session: AsyncSession):
            test_model = TransactionTestModel(
                tenant_id=self.test_tenant_id,
                name="Test Model",
                data="Test data"
            )
            session.add(test_model)
            await session.commit()  # This will fail
            return "success"
        
        # Execute function and expect rollback
        with pytest.raises(DatabaseTransactionError) as exc_info:
            await test_function(mock_session)
        
        # Verify rollback was called
        mock_session.rollback.assert_called_once()
        
        # Verify commit was attempted
        assert mock_session.commit.call_count == 1
        
        return {
            "test_name": "transaction_rollback_on_disconnection_error",
            "rollback_called": mock_session.rollback.called,
            "commit_attempted": mock_session.commit.call_count == 1,
            "exception_type": type(exc_info.value).__name__,
            "passed": True
        }
    
    async def test_transaction_success_normal_flow(self):
        """
        Test that transactions work normally when no network issues occur.
        """
        # Mock database session
        mock_session = AsyncMock()
        mock_session.in_transaction.return_value = False
        
        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_session.begin.return_value = mock_transaction
        mock_transaction.__aenter__ = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock()
        
        # Normal commit (no exception)
        mock_session.commit.return_value = None
        
        @transactional
        async def test_function(session: AsyncSession):
            test_model = TransactionTestModel(
                tenant_id=self.test_tenant_id,
                name="Test Model",
                data="Test data"
            )
            session.add(test_model)
            await session.commit()
            return "success"
        
        # Execute function successfully
        result = await test_function(mock_session)
        
        # Verify success
        assert result == "success"
        
        # Verify commit was called
        assert mock_session.commit.call_count == 1
        
        # Verify rollback was NOT called
        assert not mock_session.rollback.called
        
        return {
            "test_name": "transaction_success_normal_flow",
            "result": result,
            "commit_called": mock_session.commit.call_count == 1,
            "rollback_not_called": not mock_session.rollback.called,
            "passed": True
        }
    
    async def test_transaction_rollback_failure_handling(self):
        """
        Test that rollback failures are properly handled and logged.
        """
        # Mock database session
        mock_session = AsyncMock()
        mock_session.in_transaction.return_value = False
        
        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_session.begin.return_value = mock_transaction
        mock_transaction.__aenter__ = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock()
        
        # Simulate network failure during commit
        mock_session.commit.side_effect = OperationalError(
            "connection to server failed",
            None,
            None
        )
        
        # Simulate rollback failure
        mock_session.rollback.side_effect = Exception("Rollback failed")
        
        @transactional
        async def test_function(session: AsyncSession):
            test_model = TransactionTestModel(
                tenant_id=self.test_tenant_id,
                name="Test Model",
                data="Test data"
            )
            session.add(test_model)
            await session.commit()  # This will fail
            return "success"
        
        # Execute function and expect rollback failure to be handled
        with pytest.raises(DatabaseTransactionError) as exc_info:
            await test_function(mock_session)
        
        # Verify both commit and rollback were attempted
        assert mock_session.commit.call_count == 1
        assert mock_session.rollback.call_count == 1
        
        # Verify the exception is still raised
        assert "network failure" in str(exc_info.value)
        
        return {
            "test_name": "transaction_rollback_failure_handling",
            "commit_attempted": mock_session.commit.call_count == 1,
            "rollback_attempted": mock_session.rollback.call_count == 1,
            "exception_raised": True,
            "passed": True
        }
    
    async def run_all_tests(self):
        """Run all transaction tests and return comprehensive results."""
        await self.setup()
        
        try:
            # Run all tests
            results = []
            results.append(await self.test_transaction_rollback_on_network_failure())
            results.append(await self.test_transaction_rollback_on_disconnection_error())
            results.append(await self.test_transaction_success_normal_flow())
            results.append(await self.test_transaction_rollback_failure_handling())
            
            # Calculate overall results
            total_tests = len(results)
            passed_tests = sum(1 for result in results if result.get("passed", False))
            
            overall_result = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests) * 100,
                "test_results": results,
                "overall_passed": passed_tests == total_tests
            }
            
            return overall_result
            
        finally:
            await self.teardown()


# Pytest test functions
@pytest.mark.asyncio
async def test_transaction_rollback_on_network_failure():
    """Test V-007: Transaction Rollback Failure."""
    test_suite = TransactionTestSuite()
    result = await test_suite.test_transaction_rollback_on_network_failure()
    
    assert result["passed"], f"Transaction rollback test failed: {result}"
    assert result["rollback_called"], "Rollback was not called"
    assert result["commit_attempted"], "Commit was not attempted"


@pytest.mark.asyncio
async def test_transaction_rollback_on_disconnection_error():
    """Test V-007: Transaction Rollback on DisconnectionError."""
    test_suite = TransactionTestSuite()
    result = await test_suite.test_transaction_rollback_on_disconnection_error()
    
    assert result["passed"], f"Transaction rollback test failed: {result}"
    assert result["rollback_called"], "Rollback was not called"
    assert result["commit_attempted"], "Commit was not attempted"


@pytest.mark.asyncio
async def test_transaction_success_normal_flow():
    """Test that transactions work normally when no network issues occur."""
    test_suite = TransactionTestSuite()
    result = await test_suite.test_transaction_success_normal_flow()
    
    assert result["passed"], f"Transaction success test failed: {result}"
    assert result["result"] == "success", "Transaction did not succeed"
    assert result["rollback_not_called"], "Rollback was called unnecessarily"


@pytest.mark.asyncio
async def test_transaction_rollback_failure_handling():
    """Test that rollback failures are properly handled."""
    test_suite = TransactionTestSuite()
    result = await test_suite.test_transaction_rollback_failure_handling()
    
    assert result["passed"], f"Transaction rollback failure test failed: {result}"
    assert result["commit_attempted"], "Commit was not attempted"
    assert result["rollback_attempted"], "Rollback was not attempted"


@pytest.mark.asyncio
async def test_comprehensive_transaction_testing():
    """Run comprehensive transaction testing suite."""
    test_suite = TransactionTestSuite()
    results = await test_suite.run_all_tests()
    
    assert results["overall_passed"], f"Transaction testing suite failed: {results}"
    assert results["success_rate"] == 100.0, f"Not all tests passed: {results['success_rate']}%"
