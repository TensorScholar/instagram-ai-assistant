"""
Shopify API Integration Module

This module provides comprehensive integration with Shopify's REST and GraphQL APIs
for fetching product data, handling webhooks, and managing tenant-specific data.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timezone
import aiohttp
import json
from dataclasses import dataclass
from enum import Enum

from ..schemas.events import ProductIngested, TenantCreated
from ..schemas.models import Product, Tenant
from ..utils.security import TenantSecurity, InputValidator

logger = logging.getLogger(__name__)


class ShopifyAPIVersion(Enum):
    """Supported Shopify API versions"""
    REST_2023_10 = "2023-10"
    REST_2024_01 = "2024-01"
    GRAPHQL_2023_10 = "2023-10"


@dataclass
class ShopifyConfig:
    """Configuration for Shopify API integration"""
    shop_domain: str
    access_token: str
    api_version: ShopifyAPIVersion = ShopifyAPIVersion.REST_2024_01
    webhook_secret: Optional[str] = None
    rate_limit_per_second: int = 2
    max_retries: int = 3
    timeout_seconds: int = 30


class ShopifyAPIError(Exception):
    """Custom exception for Shopify API errors"""
    pass


class ShopifyRateLimitError(ShopifyAPIError):
    """Exception raised when Shopify rate limits are exceeded"""
    pass


class ShopifyProductFetcher:
    """
    Handles fetching product data from Shopify using both REST and GraphQL APIs.
    Supports pagination, rate limiting, and tenant-aware data processing.
    """
    
    def __init__(self, config: ShopifyConfig, tenant_id: str):
        self.config = config
        self.tenant_id = tenant_id
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = asyncio.Semaphore(config.rate_limit_per_second)
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
        
    async def connect(self) -> None:
        """Initialize HTTP session with proper headers"""
        headers = {
            'X-Shopify-Access-Token': self.config.access_token,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Aura-Platform/1.0'
        }
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
        )
        
        logger.info(f"Connected to Shopify API for tenant {self.tenant_id}")
        
    async def disconnect(self) -> None:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.info(f"Disconnected from Shopify API for tenant {self.tenant_id}")
            
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with rate limiting and error handling
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            ShopifyAPIError: For API errors
            ShopifyRateLimitError: For rate limit exceeded
        """
        async with self.rate_limiter:
            url = f"https://{self.config.shop_domain}/admin/api/{self.config.api_version.value}/{endpoint}"
            
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.request(method, url, **kwargs) as response:
                        # Handle rate limiting
                        if response.status == 429:
                            retry_after = int(response.headers.get('Retry-After', 1))
                            logger.warning(f"Rate limited, waiting {retry_after} seconds")
                            await asyncio.sleep(retry_after)
                            continue
                            
                        # Handle other HTTP errors
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(f"Shopify API error {response.status}: {error_text}")
                            raise ShopifyAPIError(f"API error {response.status}: {error_text}")
                            
                        # Parse response
                        data = await response.json()
                        return data
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Request timeout (attempt {attempt + 1})")
                    if attempt == self.config.max_retries - 1:
                        raise ShopifyAPIError("Request timeout after retries")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
                except aiohttp.ClientError as e:
                    logger.error(f"Client error: {e}")
                    if attempt == self.config.max_retries - 1:
                        raise ShopifyAPIError(f"Client error: {e}")
                    await asyncio.sleep(2 ** attempt)
                    
    async def get_shop_info(self) -> Dict[str, Any]:
        """Get shop information"""
        data = await self._make_request('GET', 'shop.json')
        return data.get('shop', {})
        
    async def get_products(self, limit: int = 250, since_id: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Fetch products with pagination support
        
        Args:
            limit: Number of products per page (max 250)
            since_id: Fetch products with ID greater than this value
            
        Yields:
            Product data dictionaries
        """
        params = {'limit': min(limit, 250)}
        if since_id:
            params['since_id'] = since_id
            
        while True:
            data = await self._make_request('GET', 'products.json', params=params)
            products = data.get('products', [])
            
            if not products:
                break
                
            for product in products:
                yield product
                
            # Check for pagination
            if len(products) < params['limit']:
                break
                
            # Update since_id for next page
            params['since_id'] = products[-1]['id']
            
    async def get_product_by_id(self, product_id: int) -> Dict[str, Any]:
        """Get specific product by ID"""
        data = await self._make_request('GET', f'products/{product_id}.json')
        return data.get('product', {})
        
    async def get_product_variants(self, product_id: int) -> List[Dict[str, Any]]:
        """Get variants for a specific product"""
        data = await self._make_request('GET', f'products/{product_id}/variants.json')
        return data.get('variants', [])
        
    async def get_product_images(self, product_id: int) -> List[Dict[str, Any]]:
        """Get images for a specific product"""
        data = await self._make_request('GET', f'products/{product_id}/images.json')
        return data.get('images', [])
        
    async def get_collections(self, limit: int = 250) -> AsyncGenerator[Dict[str, Any], None]:
        """Fetch product collections"""
        params = {'limit': min(limit, 250)}
        
        while True:
            data = await self._make_request('GET', 'collections.json', params=params)
            collections = data.get('collections', [])
            
            if not collections:
                break
                
            for collection in collections:
                yield collection
                
            if len(collections) < params['limit']:
                break
                
            params['since_id'] = collections[-1]['id']


class ShopifyWebhookHandler:
    """
    Handles Shopify webhooks for real-time data updates
    """
    
    def __init__(self, config: ShopifyConfig, tenant_id: str):
        self.config = config
        self.tenant_id = tenant_id
        self.security = TenantSecurity()
        
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify Shopify webhook signature
        
        Args:
            payload: Raw webhook payload
            signature: X-Shopify-Hmac-Sha256 header value
            
        Returns:
            True if signature is valid
        """
        if not self.config.webhook_secret:
            logger.warning("No webhook secret configured, skipping verification")
            return True
            
        import hmac
        import hashlib
        
        expected_signature = hmac.new(
            self.config.webhook_secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
        
    async def handle_product_created(self, product_data: Dict[str, Any]) -> ProductIngested:
        """Handle product created webhook"""
        return await self._process_product_data(product_data, 'created')
        
    async def handle_product_updated(self, product_data: Dict[str, Any]) -> ProductIngested:
        """Handle product updated webhook"""
        return await self._process_product_data(product_data, 'updated')
        
    async def handle_product_deleted(self, product_data: Dict[str, Any]) -> ProductIngested:
        """Handle product deleted webhook"""
        return await self._process_product_data(product_data, 'deleted')
        
    async def _process_product_data(self, product_data: Dict[str, Any], action: str) -> ProductIngested:
        """Process product data and create ProductIngested event"""
        # Validate and sanitize product data
        validator = InputValidator()
        sanitized_data = validator.sanitize_product_data(product_data)
        
        # Create ProductIngested event
        event = ProductIngested(
            tenant_id=self.tenant_id,
            product_id=str(sanitized_data['id']),
            product_data=sanitized_data,
            action=action,
            timestamp=datetime.now(timezone.utc),
            source='shopify_webhook'
        )
        
        logger.info(f"Processed {action} webhook for product {sanitized_data['id']} (tenant {self.tenant_id})")
        return event


class ShopifyDataProcessor:
    """
    Processes raw Shopify data into standardized format for storage
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.validator = InputValidator()
        
    def process_product(self, raw_product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw Shopify product data into standardized format
        
        Args:
            raw_product: Raw product data from Shopify API
            
        Returns:
            Processed product data
        """
        # Extract core product information
        processed = {
            'tenant_id': self.tenant_id,
            'external_id': str(raw_product['id']),
            'title': self.validator.sanitize_text(raw_product.get('title', '')),
            'description': self.validator.sanitize_text(raw_product.get('body_html', '')),
            'handle': raw_product.get('handle', ''),
            'vendor': raw_product.get('vendor', ''),
            'product_type': raw_product.get('product_type', ''),
            'tags': raw_product.get('tags', '').split(',') if raw_product.get('tags') else [],
            'status': raw_product.get('status', 'active'),
            'created_at': self._parse_datetime(raw_product.get('created_at')),
            'updated_at': self._parse_datetime(raw_product.get('updated_at')),
            'published_at': self._parse_datetime(raw_product.get('published_at')),
        }
        
        # Process variants
        variants = raw_product.get('variants', [])
        if variants:
            processed['variants'] = self._process_variants(variants)
            processed['price'] = variants[0].get('price', '0.00')
            processed['compare_at_price'] = variants[0].get('compare_at_price')
            processed['sku'] = variants[0].get('sku', '')
            processed['inventory_quantity'] = variants[0].get('inventory_quantity', 0)
            
        # Process images
        images = raw_product.get('images', [])
        if images:
            processed['images'] = self._process_images(images)
            processed['featured_image'] = images[0].get('src') if images else None
            
        # Process options
        options = raw_product.get('options', [])
        if options:
            processed['options'] = self._process_options(options)
            
        # Add metadata
        processed['metadata'] = {
            'source': 'shopify',
            'raw_data': raw_product,
            'processed_at': datetime.now(timezone.utc).isoformat()
        }
        
        return processed
        
    def _process_variants(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process product variants"""
        processed_variants = []
        
        for variant in variants:
            processed_variant = {
                'id': str(variant['id']),
                'title': variant.get('title', ''),
                'price': variant.get('price', '0.00'),
                'compare_at_price': variant.get('compare_at_price'),
                'sku': variant.get('sku', ''),
                'barcode': variant.get('barcode', ''),
                'inventory_quantity': variant.get('inventory_quantity', 0),
                'weight': variant.get('weight', 0),
                'weight_unit': variant.get('weight_unit', 'kg'),
                'requires_shipping': variant.get('requires_shipping', True),
                'taxable': variant.get('taxable', True),
                'position': variant.get('position', 1),
                'option_values': variant.get('option_values', [])
            }
            processed_variants.append(processed_variant)
            
        return processed_variants
        
    def _process_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process product images"""
        processed_images = []
        
        for image in images:
            processed_image = {
                'id': str(image['id']),
                'src': image.get('src', ''),
                'alt': image.get('alt', ''),
                'width': image.get('width', 0),
                'height': image.get('height', 0),
                'position': image.get('position', 1),
                'variant_ids': [str(vid) for vid in image.get('variant_ids', [])]
            }
            processed_images.append(processed_image)
            
        return processed_images
        
    def _process_options(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process product options"""
        processed_options = []
        
        for option in options:
            processed_option = {
                'id': str(option['id']),
                'name': option.get('name', ''),
                'position': option.get('position', 1),
                'values': option.get('values', [])
            }
            processed_options.append(processed_option)
            
        return processed_options
        
    def _parse_datetime(self, datetime_str: Optional[str]) -> Optional[datetime]:
        """Parse Shopify datetime string"""
        if not datetime_str:
            return None
            
        try:
            # Shopify uses ISO format with timezone
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            logger.warning(f"Failed to parse datetime: {datetime_str}")
            return None


class ShopifyIntegrationManager:
    """
    Main manager class for Shopify integration
    """
    
    def __init__(self, config: ShopifyConfig, tenant_id: str):
        self.config = config
        self.tenant_id = tenant_id
        self.fetcher = ShopifyProductFetcher(config, tenant_id)
        self.webhook_handler = ShopifyWebhookHandler(config, tenant_id)
        self.processor = ShopifyDataProcessor(tenant_id)
        
    async def fetch_all_products(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Fetch all products for the tenant"""
        async with self.fetcher:
            async for product in self.fetcher.get_products():
                processed_product = self.processor.process_product(product)
                yield processed_product
                
    async def fetch_product_by_id(self, product_id: int) -> Dict[str, Any]:
        """Fetch specific product by ID"""
        async with self.fetcher:
            raw_product = await self.fetcher.get_product_by_id(product_id)
            return self.processor.process_product(raw_product)
            
    async def handle_webhook(self, payload: bytes, signature: str, topic: str) -> Optional[ProductIngested]:
        """Handle incoming webhook"""
        # Verify signature
        if not self.webhook_handler.verify_webhook_signature(payload, signature):
            logger.error("Invalid webhook signature")
            return None
            
        # Parse payload
        try:
            data = json.loads(payload.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON payload: {e}")
            return None
            
        # Route to appropriate handler
        if topic == 'products/create':
            return await self.webhook_handler.handle_product_created(data)
        elif topic == 'products/update':
            return await self.webhook_handler.handle_product_updated(data)
        elif topic == 'products/delete':
            return await self.webhook_handler.handle_product_deleted(data)
        else:
            logger.warning(f"Unhandled webhook topic: {topic}")
            return None
