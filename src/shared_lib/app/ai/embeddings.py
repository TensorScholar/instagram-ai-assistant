"""
Vector Embedding Generation Module

This module handles the generation of vector embeddings for product data
using various embedding models and techniques.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
import json
import hashlib
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Import embedding libraries
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..utils.security import InputValidator

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Supported embedding models"""
    OPENAI_TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    OPENAI_TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    OPENAI_TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    SENTENCE_TRANSFORMERS_ALL_MPNET = "all-mpnet-base-v2"
    SENTENCE_TRANSFORMERS_ALL_MINILM = "all-MiniLM-L6-v2"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model: EmbeddingModel
    api_key: Optional[str] = None
    batch_size: int = 100
    max_retries: int = 3
    timeout_seconds: int = 30
    dimensions: Optional[int] = None  # For OpenAI models


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embeddings: List[List[float]]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class EmbeddingGenerator:
    """
    Generates vector embeddings for product data using various models
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.validator = InputValidator()
        self.model_instance = None
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize the embedding model"""
        try:
            if self.config.model.value.startswith('text-embedding'):
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI library not available")
                if not self.config.api_key:
                    raise ValueError("OpenAI API key required")
                    
                openai.api_key = self.config.api_key
                logger.info(f"Initialized OpenAI embedding model: {self.config.model.value}")
                
            elif self.config.model.value.startswith('all-'):
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError("Sentence Transformers library not available")
                    
                self.model_instance = SentenceTransformer(self.config.model.value)
                logger.info(f"Initialized Sentence Transformers model: {self.config.model.value}")
                
            else:
                raise ValueError(f"Unsupported embedding model: {self.config.model.value}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
            
    async def generate_product_embeddings(self, product_data: Dict[str, Any]) -> List[float]:
        """
        Generate embeddings for a single product
        
        Args:
            product_data: Product data dictionary
            
        Returns:
            List of embedding vectors
        """
        # Extract text content for embedding
        text_content = self._extract_text_content(product_data)
        
        # Generate embeddings
        if self.config.model.value.startswith('text-embedding'):
            embeddings = await self._generate_openai_embeddings([text_content])
        else:
            embeddings = await self._generate_sentence_transformer_embeddings([text_content])
            
        return embeddings[0] if embeddings else []
        
    async def generate_batch_embeddings(self, products: List[Dict[str, Any]]) -> EmbeddingResult:
        """
        Generate embeddings for a batch of products
        
        Args:
            products: List of product data dictionaries
            
        Returns:
            Embedding result with metadata
        """
        start_time = datetime.now()
        
        try:
            # Extract text content for all products
            text_contents = [self._extract_text_content(product) for product in products]
            
            # Generate embeddings in batches
            all_embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(text_contents), batch_size):
                batch_texts = text_contents[i:i + batch_size]
                
                if self.config.model.value.startswith('text-embedding'):
                    batch_embeddings = await self._generate_openai_embeddings(batch_texts)
                else:
                    batch_embeddings = await self._generate_sentence_transformer_embeddings(batch_texts)
                    
                all_embeddings.extend(batch_embeddings)
                
                # Add delay between batches to respect rate limits
                if i + batch_size < len(text_contents):
                    await asyncio.sleep(0.1)
                    
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create metadata
            metadata = {
                'model': self.config.model.value,
                'total_products': len(products),
                'total_embeddings': len(all_embeddings),
                'embedding_dimensions': len(all_embeddings[0]) if all_embeddings else 0,
                'processing_time_seconds': processing_time,
                'batch_size': batch_size,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return EmbeddingResult(
                embeddings=all_embeddings,
                metadata=metadata,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return EmbeddingResult(
                embeddings=[],
                metadata={'error': str(e)},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
            
    def _extract_text_content(self, product_data: Dict[str, Any]) -> str:
        """
        Extract and combine text content from product data for embedding
        
        Args:
            product_data: Product data dictionary
            
        Returns:
            Combined text content
        """
        text_parts = []
        
        # Title (most important)
        title = product_data.get('title', '')
        if title:
            text_parts.append(f"Title: {title}")
            
        # Description
        description = product_data.get('description', '')
        if description:
            # Clean HTML and limit length
            clean_description = self._clean_text_for_embedding(description)
            text_parts.append(f"Description: {clean_description}")
            
        # Product type/category
        product_type = product_data.get('product_type', '')
        if product_type:
            text_parts.append(f"Category: {product_type}")
            
        # Vendor
        vendor = product_data.get('vendor', '')
        if vendor:
            text_parts.append(f"Brand: {vendor}")
            
        # Tags
        tags = product_data.get('tags', [])
        if tags:
            tags_text = ', '.join(tags[:10])  # Limit number of tags
            text_parts.append(f"Tags: {tags_text}")
            
        # Variants information
        variants = product_data.get('variants', [])
        if variants:
            variant_info = []
            for variant in variants[:5]:  # Limit number of variants
                variant_title = variant.get('title', '')
                if variant_title and variant_title != 'Default Title':
                    variant_info.append(variant_title)
                    
            if variant_info:
                text_parts.append(f"Variants: {', '.join(variant_info)}")
                
        # Combine all text parts
        combined_text = ' | '.join(text_parts)
        
        # Limit total length to prevent token limits
        max_length = 8000  # Conservative limit for most models
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length-3] + '...'
            
        return combined_text
        
    def _clean_text_for_embedding(self, text: str) -> str:
        """Clean text for embedding generation"""
        if not text:
            return ''
            
        # Remove HTML tags
        import re
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        import html
        text = html.unescape(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Limit length
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length-3] + '...'
            
        return text
        
    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI v1 Async client"""
        embeddings: List[List[float]] = []
        client = AsyncOpenAI(api_key=self.config.api_key)  # uses env if None
        model = self.config.model.value
        for text in texts:
            for attempt in range(self.config.max_retries):
                try:
                    resp = await client.embeddings.create(
                        model=model,
                        input=text,
                        dimensions=self.config.dimensions,
                    )
                    embeddings.append(resp.data[0].embedding)
                    break
                except Exception as e:
                    logger.warning(f"OpenAI embedding attempt {attempt + 1} failed: {e}")
                    if attempt == self.config.max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
        return embeddings
        
    async def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Sentence Transformers"""
        if not self.model_instance:
            raise ValueError("Sentence Transformers model not initialized")
            
        try:
            # Generate embeddings
            embeddings = await asyncio.to_thread(
                self.model_instance.encode,
                texts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Convert to list of lists
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            logger.error(f"Failed to generate Sentence Transformers embeddings: {e}")
            raise


class EmbeddingCache:
    """
    Caches embeddings to avoid regenerating them for unchanged content
    """
    
    def __init__(self, cache_size: int = 10000):
        self.cache_size = cache_size
        self.cache = {}
        self.access_times = {}
        
    def _generate_cache_key(self, content: str, model: str) -> str:
        """Generate cache key for content and model"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{model}:{content_hash}"
        
    def get_embedding(self, content: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if available"""
        cache_key = self._generate_cache_key(content, model)
        
        if cache_key in self.cache:
            # Update access time
            self.access_times[cache_key] = datetime.now()
            return self.cache[cache_key]
            
        return None
        
    def set_embedding(self, content: str, model: str, embedding: List[float]) -> None:
        """Cache embedding"""
        cache_key = self._generate_cache_key(content, model)
        
        # Check cache size and evict if necessary
        if len(self.cache) >= self.cache_size:
            self._evict_oldest()
            
        self.cache[cache_key] = embedding
        self.access_times[cache_key] = datetime.now()
        
    def _evict_oldest(self) -> None:
        """Evict oldest cached item"""
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]


class EmbeddingSimilarityCalculator:
    """
    Calculates similarity between embeddings using various metrics
    """
    
    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
            
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    @staticmethod
    def euclidean_distance(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate Euclidean distance between two embeddings"""
        if not embedding1 or not embedding2:
            return float('inf')
            
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        return float(np.linalg.norm(vec1 - vec2))
        
    @staticmethod
    def manhattan_distance(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate Manhattan distance between two embeddings"""
        if not embedding1 or not embedding2:
            return float('inf')
            
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        return float(np.sum(np.abs(vec1 - vec2)))


class EmbeddingQualityAnalyzer:
    """
    Analyzes the quality and characteristics of generated embeddings
    """
    
    @staticmethod
    def analyze_embedding_quality(embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Analyze quality metrics for a set of embeddings
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Quality analysis results
        """
        if not embeddings:
            return {'error': 'No embeddings to analyze'}
            
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        
        # Basic statistics
        mean_embedding = np.mean(embedding_matrix, axis=0)
        std_embedding = np.std(embedding_matrix, axis=0)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = EmbeddingSimilarityCalculator.cosine_similarity(
                    embeddings[i], embeddings[j]
                )
                similarities.append(similarity)
                
        # Analyze similarity distribution
        if similarities:
            avg_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)
        else:
            avg_similarity = std_similarity = min_similarity = max_similarity = 0.0
            
        # Check for potential issues
        issues = []
        
        if avg_similarity > 0.9:
            issues.append("High average similarity - embeddings may be too similar")
            
        if std_similarity < 0.1:
            issues.append("Low similarity variance - embeddings may lack diversity")
            
        if min_similarity < -0.5:
            issues.append("Very low minimum similarity - some embeddings may be problematic")
            
        # Calculate embedding norms
        norms = [np.linalg.norm(embedding) for embedding in embeddings]
        avg_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        return {
            'total_embeddings': len(embeddings),
            'embedding_dimensions': len(embeddings[0]) if embeddings else 0,
            'mean_embedding_norm': float(np.linalg.norm(mean_embedding)),
            'std_embedding_norm': float(np.linalg.norm(std_embedding)),
            'average_pairwise_similarity': float(avg_similarity),
            'similarity_std': float(std_similarity),
            'min_similarity': float(min_similarity),
            'max_similarity': float(max_similarity),
            'average_norm': float(avg_norm),
            'norm_std': float(std_norm),
            'potential_issues': issues,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }


# Factory function for creating embedding generators
def create_embedding_generator(
    model_name: str,
    api_key: Optional[str] = None,
    batch_size: int = 100,
    dimensions: Optional[int] = None
) -> EmbeddingGenerator:
    """
    Factory function to create embedding generator with appropriate configuration
    
    Args:
        model_name: Name of the embedding model
        api_key: API key for OpenAI models
        batch_size: Batch size for processing
        dimensions: Dimensions for OpenAI models
        
    Returns:
        Configured EmbeddingGenerator instance
    """
    try:
        model = EmbeddingModel(model_name)
    except ValueError:
        raise ValueError(f"Unsupported model: {model_name}")
        
    config = EmbeddingConfig(
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        dimensions=dimensions
    )
    
    return EmbeddingGenerator(config)
