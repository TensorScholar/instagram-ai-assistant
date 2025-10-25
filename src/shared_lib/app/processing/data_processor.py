"""
Product Data Processing and Validation Module

This module handles comprehensive data processing, validation, and transformation
for product data from various e-commerce platforms.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import html
import json
from dataclasses import dataclass
from enum import Enum

from ..schemas.models import Product
from ..utils.security import InputValidator

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float
    quality_level: DataQualityLevel


@dataclass
class ProcessingMetrics:
    """Metrics for data processing operations"""
    products_processed: int
    products_validated: int
    products_failed: int
    average_quality_score: float
    processing_time_seconds: float
    errors_by_type: Dict[str, int]


class ProductDataProcessor:
    """
    Comprehensive product data processor with validation and quality assessment
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.validator = InputValidator()
        self.metrics = ProcessingMetrics(
            products_processed=0,
            products_validated=0,
            products_failed=0,
            average_quality_score=0.0,
            processing_time_seconds=0.0,
            errors_by_type={}
        )
        
    def process_product_batch(self, products: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], ProcessingMetrics]:
        """
        Process a batch of products with validation and quality assessment
        
        Args:
            products: List of raw product data
            
        Returns:
            Tuple of (processed_products, metrics)
        """
        start_time = datetime.now()
        processed_products = []
        quality_scores = []
        
        for product in products:
            try:
                # Process individual product
                processed_product = self._process_single_product(product)
                
                # Validate processed product
                validation_result = self._validate_product(processed_product)
                
                if validation_result.is_valid:
                    processed_product['validation_result'] = validation_result
                    processed_products.append(processed_product)
                    quality_scores.append(validation_result.quality_score)
                    self.metrics.products_validated += 1
                else:
                    logger.warning(f"Product validation failed: {validation_result.errors}")
                    self.metrics.products_failed += 1
                    self._update_error_metrics(validation_result.errors)
                    
            except Exception as e:
                logger.error(f"Failed to process product {product.get('id', 'unknown')}: {e}")
                self.metrics.products_failed += 1
                self._update_error_metrics([str(e)])
                
            self.metrics.products_processed += 1
            
        # Calculate final metrics
        end_time = datetime.now()
        self.metrics.processing_time_seconds = (end_time - start_time).total_seconds()
        self.metrics.average_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return processed_products, self.metrics
        
    def _process_single_product(self, raw_product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single product with comprehensive data transformation
        
        Args:
            raw_product: Raw product data from external platform
            
        Returns:
            Processed product data
        """
        processed = {
            'tenant_id': self.tenant_id,
            'external_id': str(raw_product.get('id', '')),
            'title': self._process_title(raw_product.get('title', '')),
            'description': self._process_description(raw_product.get('body_html', '')),
            'handle': self._process_handle(raw_product.get('handle', '')),
            'vendor': self._process_vendor(raw_product.get('vendor', '')),
            'product_type': self._process_product_type(raw_product.get('product_type', '')),
            'tags': self._process_tags(raw_product.get('tags', '')),
            'status': self._process_status(raw_product.get('status', 'active')),
            'created_at': self._process_datetime(raw_product.get('created_at')),
            'updated_at': self._process_datetime(raw_product.get('updated_at')),
            'published_at': self._process_datetime(raw_product.get('published_at')),
        }
        
        # Process variants
        variants = raw_product.get('variants', [])
        if variants:
            processed['variants'] = self._process_variants(variants)
            processed.update(self._extract_primary_variant_data(variants[0]))
            
        # Process images
        images = raw_product.get('images', [])
        if images:
            processed['images'] = self._process_images(images)
            processed['featured_image'] = self._extract_featured_image(images)
            
        # Process options
        options = raw_product.get('options', [])
        if options:
            processed['options'] = self._process_options(options)
            
        # Process SEO data
        processed['seo'] = self._process_seo_data(raw_product)
        
        # Process inventory data
        processed['inventory'] = self._process_inventory_data(raw_product)
        
        # Process pricing data
        processed['pricing'] = self._process_pricing_data(raw_product)
        
        # Add processing metadata
        processed['metadata'] = {
            'source': 'shopify',
            'raw_data': raw_product,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'processor_version': '1.0.0'
        }
        
        return processed
        
    def _process_title(self, title: str) -> str:
        """Process and clean product title"""
        if not title:
            return ''
            
        # Decode HTML entities
        title = html.unescape(title)
        
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title.strip())
        
        # Validate length
        if len(title) > 255:
            title = title[:252] + '...'
            
        return self.validator.sanitize_text(title)
        
    def _process_description(self, description: str) -> str:
        """Process and clean product description"""
        if not description:
            return ''
            
        # Decode HTML entities
        description = html.unescape(description)
        
        # Remove HTML tags but preserve structure
        description = re.sub(r'<[^>]+>', '', description)
        
        # Clean up whitespace
        description = re.sub(r'\s+', ' ', description.strip())
        
        # Limit length
        if len(description) > 10000:
            description = description[:9997] + '...'
            
        return self.validator.sanitize_text(description)
        
    def _process_handle(self, handle: str) -> str:
        """Process product handle (URL slug)"""
        if not handle:
            return ''
            
        # Ensure handle is URL-safe
        handle = re.sub(r'[^a-zA-Z0-9\-_]', '-', handle.lower())
        handle = re.sub(r'-+', '-', handle)
        handle = handle.strip('-')
        
        return handle[:255]  # Limit length
        
    def _process_vendor(self, vendor: str) -> str:
        """Process vendor name"""
        if not vendor:
            return ''
            
        vendor = self.validator.sanitize_text(vendor)
        return vendor[:100]  # Limit length
        
    def _process_product_type(self, product_type: str) -> str:
        """Process product type/category"""
        if not product_type:
            return 'uncategorized'
            
        product_type = self.validator.sanitize_text(product_type)
        return product_type[:100]  # Limit length
        
    def _process_tags(self, tags: Union[str, List[str]]) -> List[str]:
        """Process product tags"""
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        elif not isinstance(tags, list):
            tags = []
            
        # Clean and validate tags
        processed_tags = []
        for tag in tags:
            if tag and len(tag) <= 50:  # Limit tag length
                clean_tag = self.validator.sanitize_text(tag)
                if clean_tag and clean_tag not in processed_tags:
                    processed_tags.append(clean_tag)
                    
        return processed_tags[:20]  # Limit number of tags
        
    def _process_status(self, status: str) -> str:
        """Process product status"""
        valid_statuses = ['active', 'archived', 'draft']
        if status not in valid_statuses:
            return 'active'
        return status
        
    def _process_datetime(self, datetime_str: Optional[str]) -> Optional[datetime]:
        """Process datetime string"""
        if not datetime_str:
            return None
            
        try:
            # Handle Shopify's datetime format
            if datetime_str.endswith('Z'):
                datetime_str = datetime_str[:-1] + '+00:00'
            return datetime.fromisoformat(datetime_str)
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse datetime: {datetime_str}")
            return None
            
    def _process_variants(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process product variants"""
        processed_variants = []
        
        for variant in variants:
            processed_variant = {
                'id': str(variant.get('id', '')),
                'title': self._process_title(variant.get('title', '')),
                'price': self._process_price(variant.get('price', '0.00')),
                'compare_at_price': self._process_price(variant.get('compare_at_price')),
                'sku': self._process_sku(variant.get('sku', '')),
                'barcode': self._process_barcode(variant.get('barcode', '')),
                'inventory_quantity': self._process_inventory_quantity(variant.get('inventory_quantity', 0)),
                'weight': self._process_weight(variant.get('weight', 0)),
                'weight_unit': self._process_weight_unit(variant.get('weight_unit', 'kg')),
                'requires_shipping': bool(variant.get('requires_shipping', True)),
                'taxable': bool(variant.get('taxable', True)),
                'position': int(variant.get('position', 1)),
                'option_values': self._process_option_values(variant.get('option_values', []))
            }
            processed_variants.append(processed_variant)
            
        return processed_variants
        
    def _process_price(self, price: Union[str, float, None]) -> Optional[str]:
        """Process price value"""
        if price is None:
            return None
            
        try:
            # Convert to Decimal for precise decimal arithmetic
            decimal_price = Decimal(str(price))
            # Round to 2 decimal places
            rounded_price = decimal_price.quantize(Decimal('0.01'))
            return str(rounded_price)
        except (InvalidOperation, ValueError, TypeError):
            logger.warning(f"Invalid price value: {price}")
            return None
            
    def _process_sku(self, sku: str) -> str:
        """Process SKU"""
        if not sku:
            return ''
            
        # Clean SKU (remove special characters except hyphens and underscores)
        sku = re.sub(r'[^a-zA-Z0-9\-_]', '', sku)
        return sku[:50]  # Limit length
        
    def _process_barcode(self, barcode: str) -> str:
        """Process barcode"""
        if not barcode:
            return ''
            
        # Clean barcode (remove non-numeric characters)
        barcode = re.sub(r'[^0-9]', '', barcode)
        return barcode[:50]  # Limit length
        
    def _process_inventory_quantity(self, quantity: Union[int, str, None]) -> int:
        """Process inventory quantity"""
        if quantity is None:
            return 0
            
        try:
            return max(0, int(quantity))
        except (ValueError, TypeError):
            return 0
            
    def _process_weight(self, weight: Union[float, str, None]) -> float:
        """Process weight value"""
        if weight is None:
            return 0.0
            
        try:
            return max(0.0, float(weight))
        except (ValueError, TypeError):
            return 0.0
            
    def _process_weight_unit(self, unit: str) -> str:
        """Process weight unit"""
        valid_units = ['kg', 'g', 'lb', 'oz']
        if unit not in valid_units:
            return 'kg'
        return unit
        
    def _process_option_values(self, option_values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process option values"""
        processed_values = []
        
        for value in option_values:
            processed_value = {
                'option_id': str(value.get('option_id', '')),
                'name': self.validator.sanitize_text(value.get('name', '')),
                'value': self.validator.sanitize_text(value.get('value', ''))
            }
            processed_values.append(processed_value)
            
        return processed_values
        
    def _extract_primary_variant_data(self, primary_variant: Dict[str, Any]) -> Dict[str, Any]:
        """Extract primary variant data for main product fields"""
        return {
            'price': self._process_price(primary_variant.get('price', '0.00')),
            'compare_at_price': self._process_price(primary_variant.get('compare_at_price')),
            'sku': self._process_sku(primary_variant.get('sku', '')),
            'inventory_quantity': self._process_inventory_quantity(primary_variant.get('inventory_quantity', 0)),
            'weight': self._process_weight(primary_variant.get('weight', 0)),
            'weight_unit': self._process_weight_unit(primary_variant.get('weight_unit', 'kg'))
        }
        
    def _process_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process product images"""
        processed_images = []
        
        for image in images:
            processed_image = {
                'id': str(image.get('id', '')),
                'src': self._process_image_url(image.get('src', '')),
                'alt': self.validator.sanitize_text(image.get('alt', '')),
                'width': self._process_image_dimension(image.get('width', 0)),
                'height': self._process_image_dimension(image.get('height', 0)),
                'position': int(image.get('position', 1)),
                'variant_ids': [str(vid) for vid in image.get('variant_ids', [])]
            }
            processed_images.append(processed_image)
            
        return processed_images
        
    def _process_image_url(self, url: str) -> str:
        """Process image URL"""
        if not url:
            return ''
            
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            return ''
            
        return url[:500]  # Limit URL length
        
    def _process_image_dimension(self, dimension: Union[int, str, None]) -> int:
        """Process image dimension"""
        if dimension is None:
            return 0
            
        try:
            return max(0, int(dimension))
        except (ValueError, TypeError):
            return 0
            
    def _extract_featured_image(self, images: List[Dict[str, Any]]) -> Optional[str]:
        """Extract featured image URL"""
        if not images:
            return None
            
        # Sort by position and return first image
        sorted_images = sorted(images, key=lambda x: x.get('position', 1))
        return self._process_image_url(sorted_images[0].get('src', ''))
        
    def _process_options(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process product options"""
        processed_options = []
        
        for option in options:
            processed_option = {
                'id': str(option.get('id', '')),
                'name': self.validator.sanitize_text(option.get('name', '')),
                'position': int(option.get('position', 1)),
                'values': [self.validator.sanitize_text(val) for val in option.get('values', [])]
            }
            processed_options.append(processed_option)
            
        return processed_options
        
    def _process_seo_data(self, raw_product: Dict[str, Any]) -> Dict[str, Any]:
        """Process SEO-related data"""
        return {
            'meta_title': self.validator.sanitize_text(raw_product.get('metafields_global_title_tag', '')),
            'meta_description': self.validator.sanitize_text(raw_product.get('metafields_global_description_tag', '')),
            'seo_title': self.validator.sanitize_text(raw_product.get('title', '')),
            'seo_description': self.validator.sanitize_text(raw_product.get('body_html', ''))
        }
        
    def _process_inventory_data(self, raw_product: Dict[str, Any]) -> Dict[str, Any]:
        """Process inventory-related data"""
        variants = raw_product.get('variants', [])
        total_inventory = sum(
            self._process_inventory_quantity(v.get('inventory_quantity', 0)) 
            for v in variants
        )
        
        return {
            'total_inventory': total_inventory,
            'track_inventory': any(v.get('inventory_management') == 'shopify' for v in variants),
            'allow_backorder': any(v.get('inventory_policy') == 'continue' for v in variants),
            'variant_count': len(variants)
        }
        
    def _process_pricing_data(self, raw_product: Dict[str, Any]) -> Dict[str, Any]:
        """Process pricing-related data"""
        variants = raw_product.get('variants', [])
        if not variants:
            return {'min_price': None, 'max_price': None, 'has_discount': False}
            
        prices = []
        compare_prices = []
        
        for variant in variants:
            price = self._process_price(variant.get('price'))
            compare_price = self._process_price(variant.get('compare_at_price'))
            
            if price:
                prices.append(Decimal(price))
            if compare_price:
                compare_prices.append(Decimal(compare_price))
                
        min_price = min(prices) if prices else None
        max_price = max(prices) if prices else None
        has_discount = any(compare_prices)
        
        return {
            'min_price': str(min_price) if min_price else None,
            'max_price': str(max_price) if max_price else None,
            'has_discount': has_discount,
            'price_range': min_price != max_price if min_price and max_price else False
        }
        
    def _validate_product(self, product: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive product validation
        
        Args:
            product: Processed product data
            
        Returns:
            Validation result with quality assessment
        """
        errors = []
        warnings = []
        quality_score = 100.0
        
        # Required field validation
        if not product.get('external_id'):
            errors.append("Missing external_id")
            quality_score -= 20
            
        if not product.get('title'):
            errors.append("Missing title")
            quality_score -= 15
            
        if not product.get('price'):
            errors.append("Missing price")
            quality_score -= 15
            
        # Data quality checks
        if len(product.get('title', '')) < 3:
            warnings.append("Title too short")
            quality_score -= 5
            
        if len(product.get('description', '')) < 10:
            warnings.append("Description too short")
            quality_score -= 5
            
        if not product.get('images'):
            warnings.append("No product images")
            quality_score -= 10
            
        if not product.get('tags'):
            warnings.append("No product tags")
            quality_score -= 5
            
        # Price validation
        try:
            price = Decimal(product.get('price', '0'))
            if price <= 0:
                errors.append("Invalid price (must be positive)")
                quality_score -= 10
        except (InvalidOperation, TypeError):
            errors.append("Invalid price format")
            quality_score -= 10
            
        # Inventory validation
        inventory = product.get('inventory_quantity', 0)
        if inventory < 0:
            errors.append("Invalid inventory quantity (cannot be negative)")
            quality_score -= 5
            
        # Determine quality level
        if quality_score >= 90:
            quality_level = DataQualityLevel.EXCELLENT
        elif quality_score >= 75:
            quality_level = DataQualityLevel.GOOD
        elif quality_score >= 60:
            quality_level = DataQualityLevel.FAIR
        elif quality_score >= 40:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.CRITICAL
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=max(0, quality_score),
            quality_level=quality_level
        )
        
    def _update_error_metrics(self, errors: List[str]) -> None:
        """Update error metrics by type"""
        for error in errors:
            error_type = error.split(':')[0] if ':' in error else 'general'
            self.metrics.errors_by_type[error_type] = self.metrics.errors_by_type.get(error_type, 0) + 1


class DataQualityAnalyzer:
    """
    Analyzes data quality across product batches
    """
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_batch_quality(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze data quality for a batch of products
        
        Args:
            products: List of processed products with validation results
            
        Returns:
            Quality analysis results
        """
        if not products:
            return {'error': 'No products to analyze'}
            
        total_products = len(products)
        valid_products = sum(1 for p in products if p.get('validation_result', {}).get('is_valid', False))
        
        quality_scores = [
            p.get('validation_result', {}).get('quality_score', 0) 
            for p in products 
            if p.get('validation_result')
        ]
        
        quality_levels = [
            p.get('validation_result', {}).get('quality_level', DataQualityLevel.CRITICAL)
            for p in products 
            if p.get('validation_result')
        ]
        
        # Calculate statistics
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        quality_distribution = {}
        for level in DataQualityLevel:
            quality_distribution[level.value] = quality_levels.count(level)
            
        # Error analysis
        all_errors = []
        all_warnings = []
        
        for product in products:
            validation_result = product.get('validation_result')
            if validation_result:
                all_errors.extend(validation_result.errors)
                all_warnings.extend(validation_result.warnings)
                
        error_frequency = {}
        for error in all_errors:
            error_frequency[error] = error_frequency.get(error, 0) + 1
            
        warning_frequency = {}
        for warning in all_warnings:
            warning_frequency[warning] = warning_frequency.get(warning, 0) + 1
            
        return {
            'total_products': total_products,
            'valid_products': valid_products,
            'invalid_products': total_products - valid_products,
            'validation_rate': (valid_products / total_products) * 100 if total_products > 0 else 0,
            'average_quality_score': avg_quality_score,
            'quality_distribution': quality_distribution,
            'common_errors': dict(sorted(error_frequency.items(), key=lambda x: x[1], reverse=True)[:10]),
            'common_warnings': dict(sorted(warning_frequency.items(), key=lambda x: x[1], reverse=True)[:10]),
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
