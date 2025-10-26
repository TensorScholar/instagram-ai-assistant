"""
Aura Platform - Resilient AI Processing Tests
Tests for resilient LLM client, idempotency, and message durability.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

from shared_lib.app.ai.resilient_llm import (
    ResilientGeminiClient,
    ResilientOpenAIClient,
    ResilientLLMOrchestrator,
    LLMError,
    CircuitBreakerError,
    create_resilient_llm_orchestrator,
)
from shared_lib.app.utils.idempotency import (
    IdempotencyManager,
    idempotent_task,
    IdempotencyError,
    create_idempotency_manager,
)
from shared_lib.app.utils.message_durability import (
    MessageDurabilityManager,
    create_message_durability_manager,
)


class TestResilientGeminiClient:
    """Test ResilientGeminiClient functionality."""
    
    @pytest.fixture
    def api_key(self):
        """Gemini API key for testing."""
        return "test-gemini-api-key"
    
    @pytest.fixture
    def client(self, api_key):
        """ResilientGeminiClient instance."""
        with patch('shared_lib.app.ai.resilient_llm.genai'):
            return ResilientGeminiClient(api_key=api_key)
    
    @pytest.mark.asyncio
    async def test_generate_text_success(self, client):
        """Test successful text generation."""
        prompt = "Test prompt"
        expected_response = "Test response"
        
        # Mock successful generation
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = expected_response
        mock_model.generate_content.return_value = mock_response
        client.model = mock_model
        
        # Test generation
        result = await client.generate_text(prompt)
        
        # Verify result
        assert result == expected_response
        mock_model.generate_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_text_with_system_prompt(self, client):
        """Test text generation with system prompt."""
        prompt = "Test prompt"
        system_prompt = "You are a helpful assistant."
        expected_response = "Test response"
        
        # Mock successful generation
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = expected_response
        mock_model.generate_content.return_value = mock_response
        client.model = mock_model
        
        # Test generation
        result = await client.generate_text(prompt, system_prompt=system_prompt)
        
        # Verify result
        assert result == expected_response
        
        # Verify system prompt was included
        call_args = mock_model.generate_content.call_args[0][0]
        assert system_prompt in call_args
        assert prompt in call_args
    
    @pytest.mark.asyncio
    async def test_generate_text_empty_response(self, client):
        """Test handling of empty response."""
        prompt = "Test prompt"
        
        # Mock empty response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_model.generate_content.return_value = mock_response
        client.model = mock_model
        
        # Test generation and expect error
        with pytest.raises(LLMError, match="Empty response from Gemini"):
            await client.generate_text(prompt)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failure(self, client):
        """Test circuit breaker opens after failures."""
        prompt = "Test prompt"
        
        # Mock failure
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Test error")
        client.model = mock_model
        
        # Trigger failures to open circuit breaker
        for _ in range(4):  # More than failure threshold
            try:
                await client.generate_text(prompt)
            except Exception:
                pass
        
        # Circuit breaker should be open now
        with pytest.raises(CircuitBreakerError, match="AI Subsystem Degraded"):
            await client.generate_text(prompt)


class TestResilientOpenAIClient:
    """Test ResilientOpenAIClient functionality."""
    
    @pytest.fixture
    def api_key(self):
        """OpenAI API key for testing."""
        return "test-openai-api-key"
    
    @pytest.fixture
    def client(self, api_key):
        """ResilientOpenAIClient instance."""
        with patch('shared_lib.app.ai.resilient_llm.openai'):
            return ResilientOpenAIClient(api_key=api_key)
    
    @pytest.mark.asyncio
    async def test_generate_text_success(self, client):
        """Test successful text generation."""
        prompt = "Test prompt"
        expected_response = "Test response"
        
        # Mock successful generation
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = expected_response
        mock_response.choices = [mock_choice]
        
        client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test generation
        result = await client.generate_text(prompt)
        
        # Verify result
        assert result == expected_response
        client.client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_text_with_system_prompt(self, client):
        """Test text generation with system prompt."""
        prompt = "Test prompt"
        system_prompt = "You are a helpful assistant."
        expected_response = "Test response"
        
        # Mock successful generation
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = expected_response
        mock_response.choices = [mock_choice]
        
        client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test generation
        result = await client.generate_text(prompt, system_prompt=system_prompt)
        
        # Verify result
        assert result == expected_response
        
        # Verify messages structure
        call_args = client.client.chat.completions.create.call_args[1]
        messages = call_args['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == system_prompt
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == prompt


class TestResilientLLMOrchestrator:
    """Test ResilientLLMOrchestrator functionality."""
    
    @pytest.fixture
    def primary_client(self):
        """Primary LLM client (Gemini)."""
        with patch('shared_lib.app.ai.resilient_llm.genai'):
            return ResilientGeminiClient(api_key="test-gemini-key")
    
    @pytest.fixture
    def fallback_client(self):
        """Fallback LLM client (OpenAI)."""
        with patch('shared_lib.app.ai.resilient_llm.openai'):
            return ResilientOpenAIClient(api_key="test-openai-key")
    
    @pytest.fixture
    def orchestrator(self, primary_client, fallback_client):
        """ResilientLLMOrchestrator instance."""
        return ResilientLLMOrchestrator(
            primary_client=primary_client,
            fallback_client=fallback_client,
        )
    
    @pytest.mark.asyncio
    async def test_generate_text_primary_success(self, orchestrator):
        """Test successful generation with primary client."""
        prompt = "Test prompt"
        expected_response = "Primary response"
        
        # Mock primary client success
        orchestrator.primary_client.generate_text = AsyncMock(return_value=expected_response)
        
        # Test generation
        result = await orchestrator.generate_text(prompt)
        
        # Verify result
        assert result == expected_response
        orchestrator.primary_client.generate_text.assert_called_once_with(
            prompt=prompt,
            system_prompt=None,
        )
    
    @pytest.mark.asyncio
    async def test_generate_text_fallback_success(self, orchestrator):
        """Test fallback to secondary client when primary fails."""
        prompt = "Test prompt"
        expected_response = "Fallback response"
        
        # Mock primary client failure
        orchestrator.primary_client.generate_text = AsyncMock(
            side_effect=CircuitBreakerError("Primary failed")
        )
        
        # Mock fallback client success
        orchestrator.fallback_client.generate_text = AsyncMock(return_value=expected_response)
        
        # Test generation
        result = await orchestrator.generate_text(prompt)
        
        # Verify result
        assert result == expected_response
        orchestrator.fallback_client.generate_text.assert_called_once_with(
            prompt=prompt,
            system_prompt=None,
        )
    
    @pytest.mark.asyncio
    async def test_generate_text_both_fail(self, orchestrator):
        """Test when both primary and fallback fail."""
        prompt = "Test prompt"
        
        # Mock both clients failure
        orchestrator.primary_client.generate_text = AsyncMock(
            side_effect=CircuitBreakerError("Primary failed")
        )
        orchestrator.fallback_client.generate_text = AsyncMock(
            side_effect=CircuitBreakerError("Fallback failed")
        )
        
        # Test generation and expect error
        with pytest.raises(LLMError, match="AI Subsystem Degraded"):
            await orchestrator.generate_text(prompt)
    
    @pytest.mark.asyncio
    async def test_health_check(self, orchestrator):
        """Test health check functionality."""
        # Mock health responses
        orchestrator.primary_client.generate_text = AsyncMock(return_value="test")
        orchestrator.fallback_client.generate_text = AsyncMock(return_value="test")
        
        # Test health check
        health = await orchestrator.health_check()
        
        # Verify health status
        assert health["status"] == "healthy"
        assert "primary_client" in health
        assert "fallback_client" in health


class TestIdempotencyManager:
    """Test IdempotencyManager functionality."""
    
    @pytest.fixture
    def redis_client(self):
        """Mock Redis client."""
        return AsyncMock()
    
    @pytest.fixture
    def idempotency_manager(self, redis_client):
        """IdempotencyManager instance."""
        return IdempotencyManager(redis_client)
    
    @pytest.mark.asyncio
    async def test_check_and_set_new_operation(self, idempotency_manager):
        """Test checking and setting for new operation."""
        operation_id = "test-operation-123"
        
        # Mock Redis SET with NX returning True (new operation)
        idempotency_manager.redis_client.set.return_value = True
        
        # Test check and set
        result = await idempotency_manager.check_and_set(operation_id)
        
        # Verify result
        assert result is True
        idempotency_manager.redis_client.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_and_set_duplicate_operation(self, idempotency_manager):
        """Test checking and setting for duplicate operation."""
        operation_id = "test-operation-123"
        
        # Mock Redis SET with NX returning False (duplicate operation)
        idempotency_manager.redis_client.set.return_value = False
        
        # Test check and set
        result = await idempotency_manager.check_and_set(operation_id)
        
        # Verify result
        assert result is False
        idempotency_manager.redis_client.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_metadata(self, idempotency_manager):
        """Test getting metadata for operation."""
        operation_id = "test-operation-123"
        expected_metadata = {"operation_id": operation_id, "timestamp": 1234567890}
        
        # Mock Redis GET
        idempotency_manager.redis_client.get.return_value = '{"operation_id": "test-operation-123", "timestamp": 1234567890}'
        
        # Test get metadata
        result = await idempotency_manager.get_metadata(operation_id)
        
        # Verify result
        assert result == expected_metadata
        idempotency_manager.redis_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_key(self, idempotency_manager):
        """Test deleting idempotency key."""
        operation_id = "test-operation-123"
        
        # Mock Redis DELETE
        idempotency_manager.redis_client.delete.return_value = 1
        
        # Test delete key
        result = await idempotency_manager.delete_key(operation_id)
        
        # Verify result
        assert result is True
        idempotency_manager.redis_client.delete.assert_called_once()


class TestIdempotentTaskDecorator:
    """Test idempotent task decorator."""
    
    @pytest.fixture
    def redis_client(self):
        """Mock Redis client."""
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_idempotent_task_new_operation(self, redis_client):
        """Test idempotent task decorator with new operation."""
        
        @idempotent_task(key_arg='message_id', ttl=3600)
        async def test_task(message_id: str, data: str):
            """Test task function."""
            return {"status": "success", "message_id": message_id, "data": data}
        
        # Set Redis client for the decorator
        test_task._redis_client = redis_client
        
        # Mock Redis SET with NX returning True (new operation)
        redis_client.set.return_value = True
        
        # Test task execution
        result = await test_task("test-message-123", "test-data")
        
        # Verify result
        assert result["status"] == "success"
        assert result["message_id"] == "test-message-123"
        assert result["data"] == "test-data"
    
    @pytest.mark.asyncio
    async def test_idempotent_task_duplicate_operation(self, redis_client):
        """Test idempotent task decorator with duplicate operation."""
        
        @idempotent_task(key_arg='message_id', ttl=3600)
        async def test_task(message_id: str, data: str):
            """Test task function."""
            return {"status": "success", "message_id": message_id, "data": data}
        
        # Set Redis client for the decorator
        test_task._redis_client = redis_client
        
        # Mock Redis SET with NX returning False (duplicate operation)
        redis_client.set.return_value = False
        
        # Test task execution
        result = await test_task("test-message-123", "test-data")
        
        # Verify result
        assert result["status"] == "duplicate"
        assert result["operation_id"] == "test-message-123"
        assert result["message"] == "Task already processed"


class TestMessageDurabilityManager:
    """Test MessageDurabilityManager functionality."""
    
    @pytest.fixture
    def durability_manager(self):
        """MessageDurabilityManager instance."""
        with patch('shared_lib.app.utils.message_durability.pika'):
            return MessageDurabilityManager()
    
    @pytest.mark.asyncio
    async def test_connect(self, durability_manager):
        """Test connecting to RabbitMQ."""
        # Mock connection and channel
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        durability_manager.connection = mock_connection
        durability_manager.channel = mock_channel
        
        # Test connection
        await durability_manager.connect()
        
        # Verify connection was established
        assert durability_manager.connection is not None
        assert durability_manager.channel is not None
    
    @pytest.mark.asyncio
    async def test_declare_durable_queue(self, durability_manager):
        """Test declaring durable queue."""
        queue_name = "test_queue"
        
        # Mock channel
        mock_channel = MagicMock()
        durability_manager.channel = mock_channel
        
        # Test queue declaration
        await durability_manager.declare_durable_queue(queue_name)
        
        # Verify queue was declared
        mock_channel.queue_declare.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_message(self, durability_manager):
        """Test publishing message with durability."""
        exchange = "test_exchange"
        routing_key = "test_key"
        message = {"test": "data"}
        
        # Mock channel
        mock_channel = MagicMock()
        durability_manager.channel = mock_channel
        
        # Test message publishing
        result = await durability_manager.publish_message(
            exchange=exchange,
            routing_key=routing_key,
            message=message,
        )
        
        # Verify result
        assert result is True
        mock_channel.basic_publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check(self, durability_manager):
        """Test health check functionality."""
        # Mock connection
        mock_connection = MagicMock()
        mock_connection.is_closed = False
        durability_manager.connection = mock_connection
        
        # Test health check
        health = await durability_manager.health_check()
        
        # Verify health status
        assert health["status"] == "healthy"
        assert health["connected"] is True


class TestIntegration:
    """Integration tests for resilient AI processing."""
    
    @pytest.fixture
    def redis_client(self):
        """Mock Redis client."""
        return AsyncMock()
    
    @pytest.fixture
    def primary_client(self):
        """Primary LLM client."""
        with patch('shared_lib.app.ai.resilient_llm.genai'):
            return ResilientGeminiClient(api_key="test-gemini-key")
    
    @pytest.fixture
    def fallback_client(self):
        """Fallback LLM client."""
        with patch('shared_lib.app.ai.resilient_llm.openai'):
            return ResilientOpenAIClient(api_key="test-openai-key")
    
    @pytest.mark.asyncio
    async def test_resilient_ai_processing_integration(self, redis_client, primary_client, fallback_client):
        """Test integration of resilient AI processing components."""
        # Create orchestrator
        orchestrator = ResilientLLMOrchestrator(
            primary_client=primary_client,
            fallback_client=fallback_client,
        )
        
        # Create idempotency manager
        idempotency_manager = IdempotencyManager(redis_client)
        
        # Mock successful operations
        primary_client.generate_text = AsyncMock(return_value="AI response")
        redis_client.set.return_value = True
        
        # Test idempotency check
        is_new = await idempotency_manager.check_and_set("test-message-123")
        assert is_new is True
        
        # Test AI generation
        response = await orchestrator.generate_text("Test prompt")
        assert response == "AI response"
        
        # Verify both components worked together
        redis_client.set.assert_called_once()
        primary_client.generate_text.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
