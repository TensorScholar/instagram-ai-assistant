"""
Aura Platform - Vector Store Operations
Vector store operations with strict tenant filtering for Milvus.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
import numpy as np

from shared_lib.app.schemas.models import Product
from shared_lib.app.db.database import get_repository

logger = logging.getLogger(__name__)


class TenantAwareVectorStore:
    """Vector store with strict tenant isolation."""
    
    def __init__(
        self,
        host: str,
        port: int,
        user: str = "root",
        password: str = "Milvus",
        embedding_dimension: int = 384,  # Default for sentence-transformers/all-MiniLM-L6-v2
    ):
        """
        Initialize vector store.
        
        Args:
            host: Milvus host
            port: Milvus port
            user: Milvus username
            password: Milvus password
            embedding_dimension: Dimension of embedding vectors
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.embedding_dimension = embedding_dimension
        
        # Connect to Milvus
        self._connect()
        
        logger.info("Vector store initialized")
    
    def _connect(self) -> None:
        """Connect to Milvus."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
            )
            logger.info("Connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _get_collection_name(self, tenant_id: UUID) -> str:
        """
        Get collection name for tenant.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            Collection name
        """
        # Use a collision-resistant naming scheme
        tenant_str = str(tenant_id).replace('-', '')
        return f"tenant_{tenant_str}_products"
    
    def _get_partition_name(self, tenant_id: UUID) -> str:
        """
        Get partition name for tenant.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            Partition name
        """
        # Use a collision-resistant naming scheme
        tenant_str = str(tenant_id).replace('-', '')
        return f"tenant_{tenant_str}_partition"
    
    def _create_collection_schema(self) -> CollectionSchema:
        """
        Create collection schema for product vectors.
        
        Returns:
            Collection schema
        """
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=100,
                is_primary=True,
            ),
            FieldSchema(
                name="tenant_id",
                dtype=DataType.VARCHAR,
                max_length=100,
            ),
            FieldSchema(
                name="product_id",
                dtype=DataType.VARCHAR,
                max_length=100,
            ),
            FieldSchema(
                name="external_id",
                dtype=DataType.VARCHAR,
                max_length=255,
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=500,
            ),
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=2000,
            ),
            FieldSchema(
                name="price",
                dtype=DataType.FLOAT,
            ),
            FieldSchema(
                name="currency",
                dtype=DataType.VARCHAR,
                max_length=10,
            ),
            FieldSchema(
                name="category",
                dtype=DataType.VARCHAR,
                max_length=255,
            ),
            FieldSchema(
                name="tags",
                dtype=DataType.VARCHAR,
                max_length=1000,
            ),
            FieldSchema(
                name="embedding_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dimension,
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Product vectors with tenant isolation",
        )
        
        return schema
    
    def create_tenant_collection(self, tenant_id: UUID) -> bool:
        """
        Create vector collection and partition for tenant.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            True if created successfully, False otherwise
        """
        try:
            collection_name = self._get_collection_name(tenant_id)
            partition_name = self._get_partition_name(tenant_id)
            
            # Check if collection already exists
            if utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                # Ensure partition exists
                collection = Collection(collection_name)
                if not collection.has_partition(partition_name):
                    collection.create_partition(partition_name)
                    logger.info(f"Created partition {partition_name} for tenant {tenant_id}")
                return True
            
            # Create collection
            schema = self._create_collection_schema()
            collection = Collection(
                name=collection_name,
                schema=schema,
            )
            
            # Create partition
            collection.create_partition(partition_name)
            
            # Create index on embedding vector
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            
            collection.create_index(
                field_name="embedding_vector",
                index_params=index_params,
            )
            
            logger.info(f"Created collection {collection_name} with partition {partition_name} for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection for tenant {tenant_id}: {e}")
            return False
    
    def add_products(
        self,
        tenant_id: UUID,
        products: List[Product],
        embeddings: List[List[float]],
    ) -> bool:
        """
        Add products to tenant's vector store.
        
        Args:
            tenant_id: The tenant ID
            products: List of products
            embeddings: List of embedding vectors
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            collection_name = self._get_collection_name(tenant_id)
            partition_name = self._get_partition_name(tenant_id)
            
            # Ensure collection and partition exist
            if not utility.has_collection(collection_name):
                self.create_tenant_collection(tenant_id)
            
            collection = Collection(collection_name)
            
            # Ensure partition exists
            if not collection.has_partition(partition_name):
                collection.create_partition(partition_name)
                logger.info(f"Created partition {partition_name} for tenant {tenant_id}")
            
            # Prepare columnar data matching schema order
            ids = []
            tenant_ids = []
            product_ids = []
            external_ids = []
            titles = []
            descriptions = []
            prices = []
            currencies = []
            categories = []
            tags_list = []
            vectors = []
            for product, embedding in zip(products, embeddings):
                ids.append(f"{tenant_id}_{product.id}")
                tenant_ids.append(str(tenant_id))
                product_ids.append(str(product.id))
                external_ids.append(product.external_id)
                titles.append(product.title)
                descriptions.append(product.description or "")
                prices.append(float(product.price or 0.0))
                currencies.append(product.currency)
                categories.append(product.category or "")
                tags_list.append(",".join(product.tags) if product.tags else "")
                vectors.append(embedding)
            
            # Insert data into partition (column-based)
            entities = [
                ids,
                tenant_ids,
                product_ids,
                external_ids,
                titles,
                descriptions,
                prices,
                currencies,
                categories,
                tags_list,
                vectors,
            ]
            collection.insert(entities, partition_name=partition_name)
            collection.flush()
            
            logger.info(f"Added {len(products)} products to vector store for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding products to vector store: {e}")
            return False
    
    def search_similar_products(
        self,
        tenant_id: UUID,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar products using vector similarity.
        
        Args:
            tenant_id: The tenant ID
            query_embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filters: Additional filters
            
        Returns:
            List of similar products with metadata
        """
        try:
            collection_name = self._get_collection_name(tenant_id)
            partition_name = self._get_partition_name(tenant_id)
            
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            collection = Collection(collection_name)
            collection.load()
            
            # Build search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }
            
            # Build filter expression (simplified since we're using partitions)
            filter_expr = None
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_parts.append(f'{key} == "{value}"')
                    else:
                        filter_parts.append(f'{key} == {value}')
                filter_expr = ' and '.join(filter_parts)
            
            # Search with partition
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding_vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                partition_names=[partition_name],  # Use partition for tenant isolation
                output_fields=[
                    "product_id",
                    "external_id",
                    "title",
                    "description",
                    "price",
                    "currency",
                    "category",
                    "tags",
                ],
            )
            
            # Process results
            similar_products = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        similar_products.append({
                            "product_id": hit.entity.get("product_id"),
                            "external_id": hit.entity.get("external_id"),
                            "title": hit.entity.get("title"),
                            "description": hit.entity.get("description"),
                            "price": hit.entity.get("price"),
                            "currency": hit.entity.get("currency"),
                            "category": hit.entity.get("category"),
                            "tags": hit.entity.get("tags", "").split(",") if hit.entity.get("tags") else [],
                            "similarity_score": float(hit.score),
                        })
            
            logger.info(f"Found {len(similar_products)} similar products for tenant {tenant_id}")
            return similar_products
            
        except Exception as e:
            logger.error(f"Error searching similar products: {e}")
            return []
    
    def update_product_embedding(
        self,
        tenant_id: UUID,
        product_id: UUID,
        new_embedding: List[float],
    ) -> bool:
        """
        Update product embedding in vector store.
        
        Args:
            tenant_id: The tenant ID
            product_id: The product ID
            new_embedding: New embedding vector
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            collection_name = self._get_collection_name(tenant_id)
            
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return False
            
            collection = Collection(collection_name)
            collection.load()
            
            # Update embedding
            collection.upsert([{
                "id": f"{tenant_id}_{product_id}",
                "tenant_id": str(tenant_id),
                "embedding_vector": new_embedding,
            }])
            
            collection.flush()
            
            logger.info(f"Updated embedding for product {product_id} in tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating product embedding: {e}")
            return False
    
    def delete_product(
        self,
        tenant_id: UUID,
        product_id: UUID,
    ) -> bool:
        """
        Delete product from vector store.
        
        Args:
            tenant_id: The tenant ID
            product_id: The product ID
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            collection_name = self._get_collection_name(tenant_id)
            
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return False
            
            collection = Collection(collection_name)
            collection.load()
            
            # Delete product
            collection.delete(f'id == "{tenant_id}_{product_id}"')
            collection.flush()
            
            logger.info(f"Deleted product {product_id} from tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting product: {e}")
            return False
    
    def get_collection_stats(self, tenant_id: UUID) -> Dict[str, Any]:
        """
        Get collection statistics for tenant.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            Collection statistics
        """
        try:
            collection_name = self._get_collection_name(tenant_id)
            
            if not utility.has_collection(collection_name):
                return {"exists": False, "count": 0}
            
            collection = Collection(collection_name)
            collection.load()
            
            stats = collection.get_stats()
            
            return {
                "exists": True,
                "count": stats.get("row_count", 0),
                "collection_name": collection_name,
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"exists": False, "count": 0, "error": str(e)}
    
    def drop_tenant_collection(self, tenant_id: UUID) -> bool:
        """
        Drop tenant's vector collection.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            True if dropped successfully, False otherwise
        """
        try:
            collection_name = self._get_collection_name(tenant_id)
            
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return True
            
            utility.drop_collection(collection_name)
            
            logger.info(f"Dropped collection {collection_name} for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping collection for tenant {tenant_id}: {e}")
            return False


# Global vector store instance
_vector_store: Optional[TenantAwareVectorStore] = None


def initialize_vector_store(
    host: str,
    port: int,
    user: str = "root",
    password: str = "Milvus",
    embedding_dimension: int = 384,
) -> TenantAwareVectorStore:
    """
    Initialize global vector store.
    
    Args:
        host: Milvus host
        port: Milvus port
        user: Milvus username
        password: Milvus password
        embedding_dimension: Embedding dimension
        
    Returns:
        The vector store instance
    """
    global _vector_store
    
    _vector_store = TenantAwareVectorStore(
        host=host,
        port=port,
        user=user,
        password=password,
        embedding_dimension=embedding_dimension,
    )
    
    logger.info("Vector store initialized")
    return _vector_store


def get_vector_store() -> TenantAwareVectorStore:
    """
    Get the global vector store.
    
    Returns:
        The vector store instance
        
    Raises:
        RuntimeError: If vector store is not initialized
    """
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized")
    return _vector_store
