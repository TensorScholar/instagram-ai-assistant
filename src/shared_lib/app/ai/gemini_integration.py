"""
Aura Platform - Google Gemini LLM Integration
Integration with Google Gemini API for AI responses with brand voice.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

import google.generativeai as genai
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field

from shared_lib.app.schemas.models import Tenant
from shared_lib.app.db.database import get_repository

logger = logging.getLogger(__name__)


class GeminiLLM(LLM):
    """LangChain-compatible Gemini LLM wrapper."""
    
    api_key: str = Field(..., description="Google Gemini API key")
    model_name: str = Field(default="gemini-1.5-pro", description="Gemini model name")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize Gemini LLM."""
        super().__init__(**kwargs)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
    
    @property
    def _llm_type(self) -> str:
        """Return LLM type."""
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call Gemini API.
        
        Args:
            prompt: The prompt to send
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional arguments
            
        Returns:
            Generated response
        """
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise


class BrandVoiceManager:
    """Manages brand voice configuration for tenants."""
    
    def __init__(self):
        """Initialize brand voice manager."""
        self.default_voice_config = {
            "tone": "friendly",
            "style": "conversational",
            "personality": "helpful",
            "greeting": "Hello! How can I help you today?",
            "closing": "Is there anything else I can help you with?",
            "product_intro": "Based on your request, here are some great options:",
            "no_results": "I couldn't find exactly what you're looking for, but I'd be happy to help you find something similar.",
        }
    
    def get_tenant_voice_config(self, tenant_id: UUID) -> Dict[str, Any]:
        """
        Get brand voice configuration for tenant.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            Brand voice configuration
        """
        try:
            # Avoid async DB calls in a sync method; return default for now.
            # Future improvement: cache brand voice per tenant populated asynchronously.
            return self.default_voice_config
        except Exception as e:
            logger.error(f"Error getting tenant voice config: {e}")
            return self.default_voice_config
    
    def create_brand_prompt(
        self,
        tenant_id: UUID,
        base_prompt: str,
        context: str = "",
        query: str = "",
    ) -> str:
        """
        Create brand-aware prompt for tenant.
        
        Args:
            tenant_id: The tenant ID
            base_prompt: Base prompt template
            context: Additional context
            query: Customer query
            
        Returns:
            Brand-aware prompt
        """
        try:
            voice_config = self.get_tenant_voice_config(tenant_id)
            tenant_name = "our store"
            
            # Create brand-aware prompt
            brand_prompt = f"""You are an AI assistant for {tenant_name}. 

Brand Voice Guidelines:
- Tone: {voice_config['tone']}
- Style: {voice_config['style']}
- Personality: {voice_config['personality']}

{base_prompt}

Context about our products:
{context}

Customer Query: {query}

Please respond in a way that reflects our brand voice and helps the customer find what they're looking for."""

            return brand_prompt
            
        except Exception as e:
            logger.error(f"Error creating brand prompt: {e}")
            return base_prompt


class AIResponseGenerator:
    """Generates AI responses with brand voice and product recommendations."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.7,
    ):
        """
        Initialize AI response generator.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model name
            temperature: Generation temperature
        """
        self.llm = GeminiLLM(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
        self.brand_voice_manager = BrandVoiceManager()
    
    def generate_response(
        self,
        tenant_id: UUID,
        query: str,
        context: str = "",
        product_recommendations: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate AI response for customer query.
        
        Args:
            tenant_id: The tenant ID
            query: Customer query
            context: Additional context
            product_recommendations: Product recommendations
            conversation_history: Previous conversation messages
            
        Returns:
            Generated response with metadata
        """
        try:
            # Create base prompt
            base_prompt = self._create_base_prompt(
                query=query,
                context=context,
                product_recommendations=product_recommendations,
                conversation_history=conversation_history,
            )
            
            # Create brand-aware prompt
            brand_prompt = self.brand_voice_manager.create_brand_prompt(
                tenant_id=tenant_id,
                base_prompt=base_prompt,
                context=context,
                query=query,
            )
            
            # Generate response
            response_text = self.llm._call(brand_prompt)
            
            # Format response
            response = {
                "response_text": response_text,
                "response_type": "text",
                "product_recommendations": product_recommendations or [],
                "confidence_score": self._calculate_confidence_score(
                    query, response_text, product_recommendations
                ),
                "processing_metadata": {
                    "model_used": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "tenant_id": str(tenant_id),
                },
            }
            
            logger.info(f"Generated AI response for tenant {tenant_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return self._create_error_response(str(e))
    
    def _create_base_prompt(
        self,
        query: str,
        context: str = "",
        product_recommendations: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Create base prompt for AI response.
        
        Args:
            query: Customer query
            context: Additional context
            product_recommendations: Product recommendations
            conversation_history: Conversation history
            
        Returns:
            Base prompt
        """
        prompt_parts = [
            "You are a helpful AI assistant for an e-commerce store. Your job is to help customers find products and answer their questions.",
        ]
        
        if conversation_history:
            prompt_parts.append("\nPrevious conversation:")
            for msg in conversation_history[-3:]:  # Last 3 messages
                prompt_parts.append(f"- {msg['role']}: {msg['content']}")
        
        if context:
            prompt_parts.append(f"\nStore context:\n{context}")
        
        if product_recommendations:
            prompt_parts.append("\nRelevant products:")
            for i, rec in enumerate(product_recommendations[:3], 1):
                prompt_parts.append(
                    f"{i}. {rec['title']} - ${rec['price']} {rec['currency']}"
                )
        
        prompt_parts.append(f"\nCustomer question: {query}")
        prompt_parts.append(
            "\nPlease provide a helpful response that addresses their question and includes relevant product recommendations if applicable."
        )
        
        return "\n".join(prompt_parts)
    
    def _calculate_confidence_score(
        self,
        query: str,
        response: str,
        product_recommendations: Optional[List[Dict[str, Any]]],
    ) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            query: Original query
            response: Generated response
            product_recommendations: Product recommendations
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            score = 0.5  # Base score
            
            # Increase score if response is substantial
            if len(response) > 50:
                score += 0.2
            
            # Increase score if product recommendations are provided
            if product_recommendations and len(product_recommendations) > 0:
                score += 0.2
            
            # Increase score if response addresses the query
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            if query_words.intersection(response_words):
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create error response.
        
        Args:
            error_message: Error message
            
        Returns:
            Error response
        """
        return {
            "response_text": "I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
            "response_type": "text",
            "product_recommendations": [],
            "confidence_score": 0.0,
            "processing_metadata": {
                "model_used": "error",
                "error": error_message,
            },
        }


# Global AI response generator instance
_ai_response_generator: Optional[AIResponseGenerator] = None


def initialize_ai_response_generator(
    api_key: str,
    model_name: str = "gemini-1.5-pro",
    temperature: float = 0.7,
) -> AIResponseGenerator:
    """
    Initialize global AI response generator.
    
    Args:
        api_key: Google Gemini API key
        model_name: Gemini model name
        temperature: Generation temperature
        
    Returns:
        The AI response generator instance
    """
    global _ai_response_generator
    
    _ai_response_generator = AIResponseGenerator(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )
    
    logger.info("AI response generator initialized")
    return _ai_response_generator


def get_ai_response_generator() -> AIResponseGenerator:
    """
    Get the global AI response generator.
    
    Returns:
        The AI response generator instance
        
    Raises:
        RuntimeError: If AI response generator is not initialized
    """
    if _ai_response_generator is None:
        raise RuntimeError("AI response generator not initialized")
    return _ai_response_generator
