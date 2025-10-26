"""
Aura Platform - Resilient LLM Client

Provides fault-tolerant integration with Large Language Models (LLMs) including
Google Gemini and OpenAI GPT. Implements retry mechanisms, circuit breakers,
and intelligent fallback patterns to ensure reliable AI-powered responses
even under adverse network conditions or API failures.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)
import pybreaker
from circuit_breaker import CircuitBreakerError
import httpx
import openai

from .embeddings import EmbeddingGenerator, EmbeddingConfig, EmbeddingModel
from ..schemas.models import Product, Conversation, Message
from ..utils.security import generate_secure_token

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception class for all LLM-related operations.
    
    This serves as the parent class for all LLM-specific exceptions,
    allowing for centralized error handling and logging.
    """
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds the configured timeout duration.
    
    This typically indicates network latency issues or LLM service
    performance degradation.
    """
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limits are exceeded.
    
    Indicates that the application has exceeded the allowed number
    of requests per minute/hour for the LLM service.
    """
    pass


class LLMServiceUnavailableError(LLMError):
    """Raised when the LLM service is temporarily unavailable.
    
    This indicates server-side issues with the LLM provider
    that prevent request processing.
    """
    pass


class ResilientGeminiClient:
    """
    Resilient Gemini client with retry, circuit breaker, and fallback patterns.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        max_retries: int = 3,
        circuit_breaker_failure_threshold: int = 3,
        circuit_breaker_recovery_timeout: int = 60,
    ):
        """
        Initialize resilient Gemini client.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            circuit_breaker_failure_threshold: Number of failures before circuit opens
            circuit_breaker_recovery_timeout: Time in seconds before circuit closes
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Configure aggressive HTTP client with separate timeouts
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,  # 5 second connection timeout
                read=10.0,    # 10 second read timeout
                write=5.0,    # 5 second write timeout
                pool=5.0,     # 5 second pool timeout
            )
        )
        
        # Configure circuit breaker with timeout exceptions
        self.circuit_breaker = pybreaker.CircuitBreaker(
            fail_max=circuit_breaker_failure_threshold,
            reset_timeout=circuit_breaker_recovery_timeout,
            expected_exception=(
                LLMError,
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.HTTPStatusError,
            ),
        )
        
        logger.info(f"ResilientGeminiClient initialized with model {model_name}")
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """
        Determine if an exception is retryable.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if exception is retryable, False otherwise
        """
        retryable_exceptions = (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.HTTPStatusError,
            LLMTimeoutError,
            LLMRateLimitError,
            LLMServiceUnavailableError,
        )
        
        return isinstance(exception, retryable_exceptions)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.HTTPStatusError,
            LLMTimeoutError,
            LLMRateLimitError,
            LLMServiceUnavailableError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )
    async def _generate_text_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with retry logic.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            LLMError: If generation fails after all retries
        """
        try:
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                **kwargs,
            )
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=generation_config,
            )
            
            if response.text:
                logger.debug(f"Generated text successfully: {len(response.text)} characters")
                return response.text
            else:
                raise LLMError("Empty response from Gemini")
                
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            
            # Convert to appropriate exception type
            if isinstance(e, httpx.TimeoutException):
                raise LLMTimeoutError(f"Gemini request timed out: {e}")
            elif isinstance(e, httpx.HTTPStatusError):
                if e.response.status_code == 429:
                    raise LLMRateLimitError(f"Gemini rate limit exceeded: {e}")
                elif e.response.status_code >= 500:
                    raise LLMServiceUnavailableError(f"Gemini service unavailable: {e}")
                else:
                    raise LLMError(f"Gemini HTTP error: {e}")
            elif isinstance(e, httpx.ConnectError):
                raise LLMServiceUnavailableError(f"Gemini connection error: {e}")
            else:
                raise LLMError(f"Gemini generation error: {e}")
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with circuit breaker protection.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            CircuitBreakerError: If circuit is open
            LLMError: If generation fails
        """
        try:
            # Use circuit breaker to protect the retry-enabled method
            return await asyncio.to_thread(
                self.circuit_breaker.call,
                self._generate_text_with_retry,
                prompt,
                system_prompt,
                **kwargs,
            )
        except pybreaker.CircuitBreakerOpenException:
            logger.error("Gemini circuit breaker is open - service degraded")
            raise CircuitBreakerError("AI Subsystem Degraded - Gemini service unavailable")
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise


class ResilientOpenAIClient:
    """
    Resilient OpenAI client with retry, circuit breaker, and fallback patterns.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        max_retries: int = 3,
        circuit_breaker_failure_threshold: int = 3,
        circuit_breaker_recovery_timeout: int = 60,
    ):
        """
        Initialize resilient OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model_name: OpenAI model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            circuit_breaker_failure_threshold: Number of failures before circuit opens
            circuit_breaker_recovery_timeout: Time in seconds before circuit closes
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Configure OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
        )
        
        # Configure circuit breaker
        self.circuit_breaker = pybreaker.CircuitBreaker(
            fail_max=circuit_breaker_failure_threshold,
            reset_timeout=circuit_breaker_recovery_timeout,
            expected_exception=LLMError,
        )
        
        logger.info(f"ResilientOpenAIClient initialized with model {model_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
            LLMTimeoutError,
            LLMRateLimitError,
            LLMServiceUnavailableError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )
    async def _generate_text_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with retry logic.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            LLMError: If generation fails after all retries
        """
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
            
            if response.choices and response.choices[0].message.content:
                logger.debug(f"Generated text successfully: {len(response.choices[0].message.content)} characters")
                return response.choices[0].message.content
            else:
                raise LLMError("Empty response from OpenAI")
                
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            
            # Convert to appropriate exception type
            if isinstance(e, openai.APITimeoutError):
                raise LLMTimeoutError(f"OpenAI request timed out: {e}")
            elif isinstance(e, openai.RateLimitError):
                raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}")
            elif isinstance(e, openai.InternalServerError):
                raise LLMServiceUnavailableError(f"OpenAI service unavailable: {e}")
            elif isinstance(e, openai.APIConnectionError):
                raise LLMServiceUnavailableError(f"OpenAI connection error: {e}")
            else:
                raise LLMError(f"OpenAI generation error: {e}")
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with circuit breaker protection.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            CircuitBreakerError: If circuit is open
            LLMError: If generation fails
        """
        try:
            # Use circuit breaker to protect the retry-enabled method
            return await asyncio.to_thread(
                self.circuit_breaker.call,
                self._generate_text_with_retry,
                prompt,
                system_prompt,
                **kwargs,
            )
        except pybreaker.CircuitBreakerOpenException:
            logger.error("OpenAI circuit breaker is open - service degraded")
            raise CircuitBreakerError("AI Subsystem Degraded - OpenAI service unavailable")
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class ResilientLLMOrchestrator:
    """
    Orchestrates multiple LLM clients with intelligent fallback.
    """
    
    def __init__(
        self,
        primary_client: ResilientGeminiClient,
        fallback_client: Optional[ResilientOpenAIClient] = None,
    ):
        """
        Initialize LLM orchestrator.
        
        Args:
            primary_client: Primary LLM client (Gemini)
            fallback_client: Optional fallback client (OpenAI)
        """
        self.primary_client = primary_client
        self.fallback_client = fallback_client
        
        logger.info("ResilientLLMOrchestrator initialized with primary and fallback clients")
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with intelligent fallback.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            LLMError: If both primary and fallback fail
        """
        # Try primary client first
        try:
            logger.debug("Attempting generation with primary client (Gemini)")
            result = await self.primary_client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            )
            logger.info("Successfully generated text with primary client")
            return result
            
        except CircuitBreakerError:
            logger.warning("Primary client circuit breaker is open, trying fallback")
        except LLMError as e:
            logger.warning(f"Primary client failed: {e}, trying fallback")
        except Exception as e:
            logger.error(f"Unexpected error with primary client: {e}, trying fallback")
        
        # Try fallback client if available
        if self.fallback_client:
            try:
                logger.debug("Attempting generation with fallback client (OpenAI)")
                result = await self.fallback_client.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs,
                )
                logger.info("Successfully generated text with fallback client")
                return result
                
            except CircuitBreakerError:
                logger.error("Fallback client circuit breaker is also open")
            except LLMError as e:
                logger.error(f"Fallback client failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error with fallback client: {e}")
        
        # Both clients failed
        logger.error("All LLM clients failed - AI subsystem degraded")
        raise LLMError("AI Subsystem Degraded - All LLM services unavailable")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of all LLM clients.
        
        Returns:
            Health status information
        """
        health_status = {
            "status": "healthy",
            "primary_client": {
                "status": "unknown",
                "circuit_breaker": "unknown",
            },
            "fallback_client": {
                "status": "unknown",
                "circuit_breaker": "unknown",
            },
        }
        
        # Check primary client
        try:
            # Simple test generation
            await self.primary_client.generate_text("test", max_tokens=1)
            health_status["primary_client"]["status"] = "healthy"
        except Exception as e:
            health_status["primary_client"]["status"] = f"unhealthy: {e}"
        
        # Check circuit breaker status
        try:
            health_status["primary_client"]["circuit_breaker"] = self.primary_client.circuit_breaker.current_state
        except Exception:
            health_status["primary_client"]["circuit_breaker"] = "unknown"
        
        # Check fallback client if available
        if self.fallback_client:
            try:
                await self.fallback_client.generate_text("test", max_tokens=1)
                health_status["fallback_client"]["status"] = "healthy"
            except Exception as e:
                health_status["fallback_client"]["status"] = f"unhealthy: {e}"
            
            try:
                health_status["fallback_client"]["circuit_breaker"] = self.fallback_client.circuit_breaker.current_state
            except Exception:
                health_status["fallback_client"]["circuit_breaker"] = "unknown"
        else:
            health_status["fallback_client"]["status"] = "not_configured"
        
        # Determine overall status
        if (health_status["primary_client"]["status"] == "healthy" or 
            health_status["fallback_client"]["status"] == "healthy"):
            health_status["status"] = "healthy"
        else:
            health_status["status"] = "degraded"
        
        return health_status


# Factory function for creating resilient LLM orchestrator
def create_resilient_llm_orchestrator(
    gemini_api_key: str,
    openai_api_key: Optional[str] = None,
    gemini_model: str = "gemini-1.5-pro",
    openai_model: str = "gpt-4",
) -> ResilientLLMOrchestrator:
    """
    Create a resilient LLM orchestrator with primary and fallback clients.
    
    Args:
        gemini_api_key: Google Gemini API key
        openai_api_key: Optional OpenAI API key
        gemini_model: Gemini model name
        openai_model: OpenAI model name
        
    Returns:
        ResilientLLMOrchestrator instance
    """
    # Create primary client
    primary_client = ResilientGeminiClient(
        api_key=gemini_api_key,
        model_name=gemini_model,
    )
    
    # Create fallback client if API key is provided
    fallback_client = None
    if openai_api_key:
        fallback_client = ResilientOpenAIClient(
            api_key=openai_api_key,
            model_name=openai_model,
        )
    
    return ResilientLLMOrchestrator(
        primary_client=primary_client,
        fallback_client=fallback_client,
    )
