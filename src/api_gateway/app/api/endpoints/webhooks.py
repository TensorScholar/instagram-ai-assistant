"""
Aura Platform - API Gateway Webhook Endpoints
FastAPI endpoints for handling Instagram webhooks and other integrations.
"""

import json
import logging
import traceback
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from shared_lib.app.schemas.events import (
    EventType,
    InstagramDirectMessageReceived,
    create_event,
)
from shared_lib.app.utils.security import (
    InputValidator,
    TenantSecurity,
    get_webhook_security,
)
from ...core.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["webhooks"])


class WebhookVerificationRequest(BaseModel):
    """Instagram webhook verification request model."""
    
    hub_mode: str = Field(..., description="The verification mode")
    hub_challenge: str = Field(..., description="The challenge string")
    hub_verify_token: str = Field(..., description="The verification token")


class InstagramWebhookPayload(BaseModel):
    """Instagram webhook payload model."""
    
    object: str = Field(..., description="The object type")
    entry: list = Field(..., description="The webhook entries")


class DirectMessageData(BaseModel):
    """Instagram direct message data model."""
    
    id: str = Field(..., description="Message ID")
    from_: Dict[str, Any] = Field(..., alias="from", description="Sender information")
    message: Optional[Dict[str, Any]] = Field(None, description="Message content")
    created_time: int = Field(..., description="Creation timestamp")


class WebhookEntry(BaseModel):
    """Instagram webhook entry model."""
    
    id: str = Field(..., description="Entry ID")
    time: int = Field(..., description="Entry timestamp")
    changes: list = Field(..., description="List of changes")


@router.get("/webhooks/instagram/verify")
async def verify_instagram_webhook(
    request: Request,
    hub_mode: str,
    hub_challenge: str,
    hub_verify_token: str,
) -> str:
    """
    Verify Instagram webhook subscription.
    
    Args:
        request: The FastAPI request object
        hub_mode: The verification mode
        hub_challenge: The challenge string
        hub_verify_token: The verification token
        
    Returns:
        The challenge string if verification succeeds
        
    Raises:
        HTTPException: If verification fails
    """
    logger.info("Instagram webhook verification request received")
    
    # Verify the token
    if hub_verify_token != settings.instagram_webhook_verify_token:
        logger.warning("Instagram webhook verification failed: invalid token")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid verification token"
        )
    
    # Verify the mode
    if hub_mode != "subscribe":
        logger.warning("Instagram webhook verification failed: invalid mode")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification mode"
        )
    
    logger.info("Instagram webhook verification successful")
    return hub_challenge


@router.post("/webhooks/instagram")
async def handle_instagram_webhook(
    request: Request,
    payload: InstagramWebhookPayload,
) -> JSONResponse:
    """
    Handle Instagram webhook events.
    
    Args:
        request: The FastAPI request object
        payload: The webhook payload
        
    Returns:
        JSON response confirming receipt
        
    Raises:
        HTTPException: If processing fails
    """
    logger.info("Instagram webhook received")
    
    try:
        # Verify webhook signature
        signature = request.headers.get("X-Hub-Signature-256", "")
        raw_body = await request.body()
        
        webhook_security = get_webhook_security()
        if not webhook_security.verify_instagram_webhook(
            raw_body.decode(), signature
        ):
            logger.warning("Instagram webhook signature verification failed")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid webhook signature"
            )
        
        # Process webhook entries
        for entry_data in payload.entry:
            entry = WebhookEntry.model_validate(entry_data)
            await _process_webhook_entry(entry)
        
        logger.info("Instagram webhook processed successfully")
        return JSONResponse(
            content={"status": "success", "message": "Webhook processed"},
            status_code=status.HTTP_200_OK
        )
        
    except HTTPException as e:
        # Re-raise HTTPException to be handled by FastAPI's exception handling
        raise e
    except Exception as e:
        logger.error(f"Error processing Instagram webhook: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


async def _process_webhook_entry(entry: WebhookEntry) -> None:
    """
    Process a webhook entry.
    
    Args:
        entry: The webhook entry to process
    """
    logger.debug(f"Processing webhook entry: {entry.id}")
    
    for change in entry.changes:
        await _process_webhook_change(change, entry)


async def _process_webhook_change(change: Dict[str, Any], entry: WebhookEntry) -> None:
    """
    Process a webhook change.
    
    Args:
        change: The webhook change to process
        entry: The webhook entry containing the change
    """
    field = change.get("field")
    value = change.get("value", {})
    
    logger.debug(f"Processing webhook change for field: {field}")
    
    if field == "messages":
        await _process_direct_message(value, entry)
    else:
        logger.info(f"Unhandled webhook field: {field}")


async def _process_direct_message(
    message_data: Dict[str, Any],
    entry: WebhookEntry,
) -> None:
    """
    Process a direct message webhook.
    
    Args:
        message_data: The message data
        entry: The webhook entry
    """
    logger.info("Processing direct message webhook")
    
    try:
        # Extract message information
        message_id = message_data.get("id")
        sender_info = message_data.get("from", {})
        message_content = message_data.get("message", {})
        created_time = message_data.get("created_time")
        
        # Validate required fields
        if not message_id or not sender_info:
            logger.warning("Invalid message data: missing required fields")
            return
        
        # Extract sender information
        sender_id = sender_info.get("id")
        sender_username = sender_info.get("username")
        
        if not sender_id:
            logger.warning("Invalid sender information: missing sender ID")
            return
        
        # Extract message content
        message_text = message_content.get("text") if message_content else None
        message_type = "text"
        
        # For now, we'll use a default tenant ID
        # In a real implementation, this would be determined by the Instagram business account
        tenant_id = UUID("00000000-0000-0000-0000-000000000001")  # Default tenant
        
        # Create the event
        event = create_event(
            EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED,
            tenant_id=tenant_id,
            instagram_user_id=sender_id,
            instagram_username=sender_username,
            message_id=message_id,
            message_text=message_text,
            message_type=message_type,
            media_urls=[],  # TODO: Extract media URLs if present
            is_business_message=True,
            source_service="api_gateway",
        )
        
        # Publish event to RabbitMQ
        await _publish_event_to_queue(event)
        
        logger.info(f"Direct message event published for message {message_id}")
        
    except Exception as e:
        logger.error(f"Error processing direct message: {e}")


async def _publish_event_to_queue(event: InstagramDirectMessageReceived) -> None:
    """
    Publish an event to the message queue.
    
    Args:
        event: The event to publish
    """
    try:
        # TODO: Implement RabbitMQ publishing
        # For now, we'll just log the event
        logger.info(f"Event published to queue: {event.event_type}")
        logger.debug(f"Event data: {event.json()}")
        
        # In Phase 1, this would publish to RabbitMQ
        # The intelligence worker would then consume this event
        
    except Exception as e:
        logger.error(f"Error publishing event to queue: {e}")
        raise


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "service": "api_gateway",
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": "2024-01-01T00:00:00Z",  # TODO: Use actual timestamp
    }


@router.get("/webhooks/instagram/test")
async def test_instagram_webhook() -> Dict[str, Any]:
    """
    Test endpoint for Instagram webhook processing.
    
    Returns:
        Test response
    """
    # Create a test event
    test_event = create_event(
        EventType.INSTAGRAM_DIRECT_MESSAGE_RECEIVED,
        tenant_id=UUID("00000000-0000-0000-0000-000000000001"),
        instagram_user_id="test_user_123",
        instagram_username="test_user",
        message_id="test_message_456",
        message_text="Hello, this is a test message!",
        message_type="text",
        media_urls=[],
        is_business_message=True,
        source_service="api_gateway",
    )
    
    # Publish test event
    await _publish_event_to_queue(test_event)
    
    return {
        "status": "success",
        "message": "Test event published",
        "event_id": str(test_event.event_id),
        "correlation_id": str(test_event.correlation_id),
    }
