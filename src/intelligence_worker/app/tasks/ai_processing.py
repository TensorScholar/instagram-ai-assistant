"""
Aura Platform - Intelligence Worker Message Processing Tasks
Celery tasks for processing Instagram messages with full AI capabilities.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from celery import current_task
from shared_lib.app.schemas.events import (
    EventType,
    InstagramDirectMessageReceived,
    AIQueryProcessed,
    AIQueryFailed,
)
from shared_lib.app.schemas.models import EventLog, Tenant, Product
from shared_lib.app.db.database import get_repository, initialize_database
from shared_lib.app.utils.security import initialize_security
from shared_lib.app.utils.rabbitmq import initialize_rabbitmq, get_event_publisher
from shared_lib.app.ai.rag_pipeline import initialize_rag_pipeline, get_rag_pipeline, get_recommendation_engine
from shared_lib.app.ai.gemini_integration import initialize_ai_response_generator, get_ai_response_generator
from shared_lib.app.ai.vector_store import initialize_vector_store, get_vector_store

from .celery_app import intelligence_task, message_task

logger = logging.getLogger(__name__)


class IntelligenceProcessor:
    """Main intelligence processor with full AI capabilities."""
    
    def __init__(self):
        """Initialize intelligence processor."""
        self.repository = None
        self.rag_pipeline = None
        self.ai_generator = None
        self.recommendation_engine = None
        self.vector_store = None
        self.event_publisher = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all AI components."""
        if self._initialized:
            return
        
        try:
            # Initialize database
            from .config import settings
            initialize_database(settings.database_url)
            self.repository = get_repository()
            
            # Initialize security
            initialize_security(
                secret_key=settings.secret_key,
                webhook_secret="test_webhook_secret",  # TODO: Get from settings
            )
            
            # Initialize RabbitMQ
            initialize_rabbitmq(
                host=settings.rabbitmq_host,
                port=settings.rabbitmq_port,
                username=settings.rabbitmq_user,
                password=settings.rabbitmq_password,
                vhost=settings.rabbitmq_vhost,
            )
            self.event_publisher = get_event_publisher()
            
            # Initialize RAG pipeline
            initialize_rag_pipeline(
                milvus_host=settings.milvus_host,
                milvus_port=settings.milvus_port,
            )
            self.rag_pipeline = get_rag_pipeline()
            self.recommendation_engine = get_recommendation_engine()
            
            # Initialize AI response generator
            initialize_ai_response_generator(
                api_key=settings.gemini_api_key,
                model_name=settings.gemini_model,
            )
            self.ai_generator = get_ai_response_generator()
            
            # Initialize vector store
            initialize_vector_store(
                host=settings.milvus_host,
                port=settings.milvus_port,
            )
            self.vector_store = get_vector_store()
            
            self._initialized = True
            logger.info("Intelligence processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize intelligence processor: {e}")
            raise
    
    def process_instagram_message(
        self,
        event: InstagramDirectMessageReceived,
    ) -> Dict[str, Any]:
        """
        Process Instagram direct message with full AI capabilities.
        
        Args:
            event: Instagram direct message event
            
        Returns:
            Processing result
        """
        try:
            if not self._initialized:
                self.initialize()
            
            logger.info(f"Processing Instagram message {event.message_id} for tenant {event.tenant_id}")
            
            # Get tenant information
            tenant = self.repository.get_by_id(Tenant, event.tenant_id, event.tenant_id)
            if not tenant:
                raise ValueError(f"Tenant {event.tenant_id} not found")
            
            # Get relevant products using RAG pipeline
            query = event.message_text or "general inquiry"
            relevant_products = self.rag_pipeline.search_products(
                tenant_id=event.tenant_id,
                query=query,
                k=5,
            )
            
            # Generate product recommendations
            recommendations = self.recommendation_engine.get_recommendations(
                tenant_id=event.tenant_id,
                query=query,
                max_recommendations=3,
            )
            
            # Generate AI response with brand voice
            ai_response = self.ai_generator.generate_response(
                tenant_id=event.tenant_id,
                query=query,
                context=self._create_context_string(relevant_products),
                product_recommendations=recommendations,
            )
            
            # Create AI query processed event
            ai_event = AIQueryProcessed(
                tenant_id=event.tenant_id,
                query_id=event.event_id,
                response_text=ai_response["response_text"],
                response_type=ai_response["response_type"],
                product_recommendations=recommendations,
                processing_time_ms=1500,  # TODO: Calculate actual processing time
                ai_model_used=ai_response["processing_metadata"]["model_used"],
                confidence_score=ai_response["confidence_score"],
                source_service="intelligence_worker",
            )
            
            # Publish AI response event
            self.event_publisher.rabbitmq_manager.publish_event(ai_event)
            
            logger.info(f"Successfully processed Instagram message {event.message_id}")
            
            return {
                "status": "success",
                "message": "Instagram message processed successfully",
                "event_id": str(event.event_id),
                "correlation_id": str(event.correlation_id),
                "ai_response": ai_response["response_text"],
                "recommendations": recommendations,
                "confidence_score": ai_response["confidence_score"],
            }
            
        except Exception as e:
            logger.error(f"Error processing Instagram message: {e}")
            
            # Create AI query failed event
            try:
                ai_failed_event = AIQueryFailed(
                    tenant_id=event.tenant_id,
                    query_id=event.event_id,
                    error_message=str(e),
                    error_code="PROCESSING_ERROR",
                    retry_count=0,
                    source_service="intelligence_worker",
                )
                self.event_publisher.rabbitmq_manager.publish_event(ai_failed_event)
            except Exception as log_error:
                logger.error(f"Error creating AI failed event: {log_error}")
            
            return {
                "status": "error",
                "message": f"Failed to process Instagram message: {e}",
                "event_id": str(event.event_id),
                "correlation_id": str(event.correlation_id),
            }
    
    def _create_context_string(self, products: List[Dict[str, Any]]) -> str:
        """
        Create context string from products.
        
        Args:
            products: List of product information
            
        Returns:
            Context string
        """
        if not products:
            return "No relevant products found."
        
        context_parts = []
        for product in products:
            context_parts.append(
                f"Product: {product['title']} - "
                f"Price: {product['price']} {product['currency']} - "
                f"Category: {product['category']}"
            )
        
        return "\n".join(context_parts)


# Global intelligence processor
_intelligence_processor: Optional[IntelligenceProcessor] = None


def get_intelligence_processor() -> IntelligenceProcessor:
    """
    Get the global intelligence processor.
    
    Returns:
        The intelligence processor instance
    """
    global _intelligence_processor
    
    if _intelligence_processor is None:
        _intelligence_processor = IntelligenceProcessor()
    
    return _intelligence_processor


@message_task(name="process_instagram_direct_message")
def process_instagram_direct_message(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an Instagram direct message event with full AI capabilities.
    
    Args:
        event_data: The Instagram direct message event data
        
    Returns:
        Processing result
    """
    logger.info("Processing Instagram direct message with AI capabilities")
    
    try:
        # Parse the event
        event = InstagramDirectMessageReceived(**event_data)
        
        # Get intelligence processor
        processor = get_intelligence_processor()
        
        # Process the message
        result = processor.process_instagram_message(event)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing Instagram direct message: {e}")
        return {
            "status": "error",
            "message": f"Failed to process Instagram direct message: {e}",
        }


@intelligence_task(name="process_ai_query")
def process_ai_query(
    query_text: str,
    tenant_id: str,
    instagram_user_id: str,
    conversation_history: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Process an AI query for a tenant with full capabilities.
    
    Args:
        query_text: The query text
        tenant_id: The tenant ID
        instagram_user_id: The Instagram user ID
        conversation_history: Previous conversation messages
        
    Returns:
        AI processing result
    """
    logger.info(f"Processing AI query for tenant {tenant_id}")
    
    try:
        # Get intelligence processor
        processor = get_intelligence_processor()
        processor.initialize()
        
        tenant_uuid = UUID(tenant_id)
        
        # Get relevant products using RAG pipeline
        relevant_products = processor.rag_pipeline.search_products(
            tenant_id=tenant_uuid,
            query=query_text,
            k=5,
        )
        
        # Generate product recommendations
        recommendations = processor.recommendation_engine.get_recommendations(
            tenant_id=tenant_uuid,
            query=query_text,
            max_recommendations=3,
        )
        
        # Generate AI response with brand voice
        ai_response = processor.ai_generator.generate_response(
            tenant_id=tenant_uuid,
            query=query_text,
            context=processor._create_context_string(relevant_products),
            product_recommendations=recommendations,
            conversation_history=conversation_history,
        )
        
        logger.info(f"AI query processed successfully for tenant {tenant_id}")
        
        return {
            "status": "success",
            "response_text": ai_response["response_text"],
            "response_type": ai_response["response_type"],
            "product_recommendations": recommendations,
            "processing_time_ms": 2000,  # TODO: Calculate actual processing time
            "ai_model_used": ai_response["processing_metadata"]["model_used"],
            "confidence_score": ai_response["confidence_score"],
        }
        
    except Exception as e:
        logger.error(f"Error processing AI query: {e}")
        
        return {
            "status": "error",
            "error_message": str(e),
            "error_code": "AI_PROCESSING_ERROR",
        }


@intelligence_task(name="generate_product_recommendations")
def generate_product_recommendations(
    query_text: str,
    tenant_id: str,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Generate product recommendations based on a query with full AI capabilities.
    
    Args:
        query_text: The query text
        tenant_id: The tenant ID
        limit: Maximum number of recommendations
        
    Returns:
        Product recommendations
    """
    logger.info(f"Generating product recommendations for tenant {tenant_id}")
    
    try:
        # Get intelligence processor
        processor = get_intelligence_processor()
        processor.initialize()
        
        tenant_uuid = UUID(tenant_id)
        
        # Generate recommendations using RAG pipeline
        recommendations = processor.recommendation_engine.get_recommendations(
            tenant_id=tenant_uuid,
            query=query_text,
            max_recommendations=limit,
        )
        
        logger.info(f"Generated {len(recommendations)} product recommendations")
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "total_count": len(recommendations),
            "query": query_text,
        }
        
    except Exception as e:
        logger.error(f"Error generating product recommendations: {e}")
        
        return {
            "status": "error",
            "error_message": str(e),
            "error_code": "RECOMMENDATION_ERROR",
        }


@intelligence_task(name="update_product_embeddings")
def update_product_embeddings(tenant_id: str) -> Dict[str, Any]:
    """
    Update product embeddings for a tenant.
    
    Args:
        tenant_id: The tenant ID
        
    Returns:
        Update result
    """
    logger.info(f"Updating product embeddings for tenant {tenant_id}")
    
    try:
        # Get intelligence processor
        processor = get_intelligence_processor()
        processor.initialize()
        
        tenant_uuid = UUID(tenant_id)
        
        # Get all products for tenant
        products = processor.repository.get_all(
            Product,
            tenant_uuid,
            limit=1000,  # TODO: Implement pagination
        )
        
        if not products:
            return {
                "status": "success",
                "message": "No products found to update",
                "updated_count": 0,
            }
        
        # TODO: Generate embeddings for products
        # For now, create dummy embeddings
        embeddings = [[0.1] * 384 for _ in products]  # 384 is the default embedding dimension
        
        # Add products to vector store
        success = processor.vector_store.add_products(
            tenant_id=tenant_uuid,
            products=products,
            embeddings=embeddings,
        )
        
        if success:
            logger.info(f"Updated embeddings for {len(products)} products")
            return {
                "status": "success",
                "message": f"Updated embeddings for {len(products)} products",
                "updated_count": len(products),
            }
        else:
            return {
                "status": "error",
                "message": "Failed to update product embeddings",
                "updated_count": 0,
            }
        
    except Exception as e:
        logger.error(f"Error updating product embeddings: {e}")
        
        return {
            "status": "error",
            "error_message": str(e),
            "error_code": "EMBEDDING_UPDATE_ERROR",
        }


# Task for logging events to database
@intelligence_task(name="log_event_to_database")
def log_event_to_database(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log an event to the database.
    
    Args:
        event_data: The event data to log
        
    Returns:
        Logging result
    """
    logger.info("Logging event to database")
    
    try:
        # Get intelligence processor
        processor = get_intelligence_processor()
        processor.initialize()
        
        # Create event log entry
        event_log = EventLog(
            tenant_id=UUID(event_data["tenant_id"]),
            event_id=UUID(event_data["event_id"]),
            event_type=event_data["event_type"],
            correlation_id=UUID(event_data["correlation_id"]),
            event_data=event_data,
            source_service=event_data.get("source_service", "intelligence_worker"),
            processed=True,
        )
        
        # Save to database
        processor.repository.create(
            EventLog,
            UUID(event_data["tenant_id"]),
            **event_log.dict(),
        )
        
        logger.info(f"Event logged: {event_data['event_type']} - {event_data['event_id']}")
        
        return {
            "status": "success",
            "message": "Event logged successfully",
            "event_id": event_data["event_id"],
        }
        
    except Exception as e:
        logger.error(f"Error logging event to database: {e}")
        
        return {
            "status": "error",
            "message": f"Failed to log event: {e}",
        }
