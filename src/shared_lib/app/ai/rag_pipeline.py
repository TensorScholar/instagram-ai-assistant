"""
Aura Platform - LangChain RAG Pipeline
Tenant-aware Retrieval Augmented Generation pipeline for AI responses.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter

from shared_lib.app.db.database import get_repository
from shared_lib.app.schemas.models import Product, Tenant

logger = logging.getLogger(__name__)


class TenantAwareRAGPipeline:
    """Tenant-aware RAG pipeline for AI responses."""
    
    def __init__(
        self,
        milvus_host: str,
        milvus_port: int,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            milvus_host: Milvus host
            milvus_port: Milvus port
            embedding_model: Embedding model name
            chunk_size: Text chunk size
            chunk_overlap: Text chunk overlap
        """
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        self.vector_store: Optional[Milvus] = None
        self.retrieval_chain: Optional[RetrievalQA] = None
        
        logger.info("RAG pipeline initialized")
    
    def _get_tenant_collection_name(self, tenant_id: UUID) -> str:
        """
        Get collection name for tenant.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            Collection name
        """
        return f"tenant_{str(tenant_id).replace('-', '_')}_products"
    
    def _create_tenant_vector_store(self, tenant_id: UUID) -> Milvus:
        """
        Create vector store for tenant.
        
        Args:
            tenant_id: The tenant ID
            
        Returns:
            Milvus vector store instance
        """
        collection_name = self._get_tenant_collection_name(tenant_id)
        
        vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={
                "host": self.milvus_host,
                "port": self.milvus_port,
            },
            collection_name=collection_name,
            drop_old=False,
        )
        
        logger.info(f"Created vector store for tenant {tenant_id}")
        return vector_store
    
    def add_products_to_vector_store(
        self,
        tenant_id: UUID,
        products: List[Product],
    ) -> None:
        """
        Add products to tenant's vector store.
        
        Args:
            tenant_id: The tenant ID
            products: List of products to add
        """
        try:
            vector_store = self._create_tenant_vector_store(tenant_id)
            
            # Prepare documents
            documents = []
            for product in products:
                # Create product text for embedding
                product_text = self._create_product_text(product)
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(product_text)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "product_id": str(product.id),
                            "tenant_id": str(tenant_id),
                            "external_id": product.external_id,
                            "title": product.title,
                            "price": product.price,
                            "currency": product.currency,
                            "category": product.category,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        },
                    )
                    documents.append(doc)
            
            # Add documents to vector store
            if documents:
                vector_store.add_documents(documents)
                logger.info(f"Added {len(documents)} product chunks to vector store for tenant {tenant_id}")
            
        except Exception as e:
            logger.error(f"Error adding products to vector store: {e}")
            raise
    
    def _create_product_text(self, product: Product) -> str:
        """
        Create text representation of product for embedding.
        
        Args:
            product: The product
            
        Returns:
            Product text
        """
        text_parts = [
            f"Product: {product.title}",
            f"Description: {product.description or 'No description available'}",
            f"Price: {product.price} {product.currency}" if product.price else "Price not available",
            f"Category: {product.category}" if product.category else "No category",
        ]
        
        if product.tags:
            text_parts.append(f"Tags: {', '.join(product.tags)}")
        
        return " | ".join(text_parts)
    
    def search_products(
        self,
        tenant_id: UUID,
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant products using vector similarity.
        
        Args:
            tenant_id: The tenant ID
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant products with metadata
        """
        try:
            vector_store = self._create_tenant_vector_store(tenant_id)
            
            # Search for similar documents
            results = vector_store.similarity_search_with_score(
                query,
                k=k,
                filter={"tenant_id": str(tenant_id)},
            )
            
            # Filter by score threshold and format results
            relevant_products = []
            for doc, score in results:
                if score >= score_threshold:
                    metadata = doc.metadata
                    relevant_products.append({
                        "product_id": metadata["product_id"],
                        "title": metadata["title"],
                        "price": metadata["price"],
                        "currency": metadata["currency"],
                        "category": metadata["category"],
                        "similarity_score": float(score),
                        "chunk_content": doc.page_content,
                    })
            
            logger.info(f"Found {len(relevant_products)} relevant products for query: {query}")
            return relevant_products
            
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []
    
    def create_retrieval_chain(
        self,
        tenant_id: UUID,
        llm,
        prompt_template: Optional[str] = None,
    ) -> RetrievalQA:
        """
        Create retrieval chain for tenant.
        
        Args:
            tenant_id: The tenant ID
            llm: Language model instance
            prompt_template: Custom prompt template
            
        Returns:
            RetrievalQA chain
        """
        try:
            vector_store = self._create_tenant_vector_store(tenant_id)
            
            # Create retriever with tenant filtering
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 5,
                    "filter": {"tenant_id": str(tenant_id)},
                }
            )
            
            # Default prompt template
            if not prompt_template:
                prompt_template = """You are an AI assistant for {tenant_name}. You help customers find products and answer questions about the store.

Use the following pieces of context to answer the customer's question. If you don't know the answer based on the context, say that you don't know and ask for more information.

Context:
{context}

Customer Question: {question}

Answer: Provide a helpful response that includes relevant product recommendations if applicable. Be friendly and maintain the brand voice of {tenant_name}."""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "tenant_name"],
            )
            
            # Create retrieval chain
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True,
            )
            
            logger.info(f"Created retrieval chain for tenant {tenant_id}")
            return chain
            
        except Exception as e:
            logger.error(f"Error creating retrieval chain: {e}")
            raise
    
    def get_tenant_context(
        self,
        tenant_id: UUID,
        query: str,
        k: int = 3,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get tenant-specific context for a query.
        
        Args:
            tenant_id: The tenant ID
            query: The query
            k: Number of relevant products to retrieve
            
        Returns:
            Tuple of context string and relevant products
        """
        try:
            # Get relevant products
            relevant_products = self.search_products(tenant_id, query, k=k)
            
            # Create context string
            context_parts = []
            for product in relevant_products:
                context_parts.append(
                    f"Product: {product['title']} - "
                    f"Price: {product['price']} {product['currency']} - "
                    f"Category: {product['category']}"
                )
            
            context = "\n".join(context_parts) if context_parts else "No relevant products found."
            
            return context, relevant_products
            
        except Exception as e:
            logger.error(f"Error getting tenant context: {e}")
            return "Error retrieving context.", []


class ProductRecommendationEngine:
    """Product recommendation engine using RAG pipeline."""
    
    def __init__(self, rag_pipeline: TenantAwareRAGPipeline):
        """
        Initialize recommendation engine.
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline
    
    def get_recommendations(
        self,
        tenant_id: UUID,
        query: str,
        max_recommendations: int = 5,
        price_range: Optional[Tuple[float, float]] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get product recommendations based on query.
        
        Args:
            tenant_id: The tenant ID
            query: Customer query
            max_recommendations: Maximum number of recommendations
            price_range: Optional price range filter
            category_filter: Optional category filter
            
        Returns:
            List of product recommendations
        """
        try:
            # Get relevant products
            relevant_products = self.rag_pipeline.search_products(
                tenant_id=tenant_id,
                query=query,
                k=max_recommendations * 2,  # Get more to filter
            )
            
            # Apply filters
            filtered_products = relevant_products
            
            if price_range:
                min_price, max_price = price_range
                filtered_products = [
                    p for p in filtered_products
                    if p["price"] and min_price <= p["price"] <= max_price
                ]
            
            if category_filter:
                filtered_products = [
                    p for p in filtered_products
                    if p["category"] and category_filter.lower() in p["category"].lower()
                ]
            
            # Limit to max recommendations
            recommendations = filtered_products[:max_recommendations]
            
            # Add recommendation metadata
            for i, rec in enumerate(recommendations):
                rec["recommendation_rank"] = i + 1
                rec["recommendation_reason"] = f"Similarity score: {rec['similarity_score']:.2f}"
            
            logger.info(f"Generated {len(recommendations)} recommendations for tenant {tenant_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []


# Global RAG pipeline instance
_rag_pipeline: Optional[TenantAwareRAGPipeline] = None
_recommendation_engine: Optional[ProductRecommendationEngine] = None


def initialize_rag_pipeline(
    milvus_host: str,
    milvus_port: int,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> TenantAwareRAGPipeline:
    """
    Initialize global RAG pipeline.
    
    Args:
        milvus_host: Milvus host
        milvus_port: Milvus port
        embedding_model: Embedding model name
        
    Returns:
        The RAG pipeline instance
    """
    global _rag_pipeline, _recommendation_engine
    
    _rag_pipeline = TenantAwareRAGPipeline(
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        embedding_model=embedding_model,
    )
    
    _recommendation_engine = ProductRecommendationEngine(_rag_pipeline)
    
    logger.info("RAG pipeline initialized")
    return _rag_pipeline


def get_rag_pipeline() -> TenantAwareRAGPipeline:
    """
    Get the global RAG pipeline.
    
    Returns:
        The RAG pipeline instance
        
    Raises:
        RuntimeError: If RAG pipeline is not initialized
    """
    if _rag_pipeline is None:
        raise RuntimeError("RAG pipeline not initialized")
    return _rag_pipeline


def get_recommendation_engine() -> ProductRecommendationEngine:
    """
    Get the global recommendation engine.
    
    Returns:
        The recommendation engine instance
        
    Raises:
        RuntimeError: If recommendation engine is not initialized
    """
    if _recommendation_engine is None:
        raise RuntimeError("Recommendation engine not initialized")
    return _recommendation_engine
