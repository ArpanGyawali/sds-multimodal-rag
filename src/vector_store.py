"""Vector store implementation using Weaviate."""

import json
import uuid
from typing import List, Dict, Any, Optional

import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
# Import Gemini embeddings  
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import settings, get_weaviate_schema
from .pdf_processor import DocumentChunk


class WeaviateVectorStore:
    """Vector store implementation using Weaviate Cloud."""
    
    def __init__(self):
        self.client = None
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model
        )
        self._connect()
    
    def _connect(self):
        """Connect to Weaviate Cloud instance."""
        try:
            self.client = weaviate.Client(
                url=settings.weaviate_url,
                auth_client_secret=weaviate.AuthApiKey(api_key=settings.weaviate_api_key),
                additional_headers={
                    "X-OpenAI-Api-Key": settings.openai_api_key
                }
            )
            
            # Test connection
            if self.client.is_ready():
                print("Successfully connected to Weaviate")
            else:
                raise Exception("Weaviate is not ready")
                
        except Exception as e:
            print(f"Failed to connect to Weaviate: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test the Weaviate connection."""
        try:
            return self.client is not None and self.client.is_ready()
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    def create_schema(self, reset: bool = False):
        """Create or update the Weaviate schema."""
        try:
            class_name = settings.weaviate_class_name
            
            # If reset is requested, delete existing schema first
            if reset:
                try:
                    self.client.schema.delete_class(class_name)
                    print(f"🗑️ Deleted existing class: {class_name}")
                except Exception as e:
                    print(f"Note: Could not delete existing class (may not exist): {e}")
            
            # Check if class already exists
            existing_schema = self.client.schema.get()
            existing_classes = [cls['class'] for cls in existing_schema.get('classes', [])]
            
            if class_name in existing_classes:
                print(f"✅ Class {class_name} already exists, skipping creation")
                return True
            
            # Create schema
            schema = get_weaviate_schema()
            self.client.schema.create_class(schema)
            print(f"✅ Created Weaviate class: {class_name}")
            return True
            
        except Exception as e:
            # Handle the specific "already exists" error gracefully
            if "already exists" in str(e).lower():
                print(f"✅ Class {settings.weaviate_class_name} already exists, continuing...")
                return True
            else:
                print(f"❌ Error creating schema: {e}")
                return False
    
    def delete_schema(self):
        """Delete the schema (useful for resets)."""
        try:
            if self.client.schema.exists(settings.weaviate_class_name):
                self.client.schema.delete_class(settings.weaviate_class_name)
                print(f"Deleted Weaviate class: {settings.weaviate_class_name}")
        except Exception as e:
            print(f"Error deleting schema: {e}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to the vector store."""
        if not chunks:
            return []
        
        # Generate embeddings for all chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Batch insert
        object_ids = []
        batch_size = 100
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            with self.client.batch as batch:
                batch.batch_size = batch_size
                
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    object_id = str(uuid.uuid4())
                    
                    # Prepare data object
                    data_object = chunk.to_dict()
                    
                    # Add to batch
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=settings.weaviate_class_name,
                        uuid=object_id,
                        vector=embedding
                    )
                    
                    object_ids.append(object_id)
        
        print(f"Added {len(chunks)} chunks to vector store")
        return object_ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        content_type: Optional[str] = None,
        document_name: Optional[str] = None
    ) -> List[Document]:
        """Perform similarity search with optional filtering."""
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Build where clause for filtering
        where_clause = {}
        if content_type:
            where_clause["content_type"] = {"equal": content_type}
        if document_name:
            if where_clause:
                where_clause = {
                    "operator": "And",
                    "operands": [
                        where_clause,
                        {"path": ["document_name"], "operator": "Equal", "valueString": document_name}
                    ]
                }
            else:
                where_clause = {"path": ["document_name"], "operator": "Equal", "valueString": document_name}
        
        # Perform search
        try:
            query_builder = (
                self.client.query
                .get(settings.weaviate_class_name, [
                    "content", "content_type", "page_number", 
                    "chunk_index", "document_name", "bbox", "metadata"
                ])
                .with_near_vector({"vector": query_embedding})
                .with_limit(k)
            )
            
            if where_clause:
                query_builder = query_builder.with_where(where_clause)
            
            results = query_builder.do()
            
            # Convert to LangChain Document format
            documents = []
            if "data" in results and "Get" in results["data"]:
                for item in results["data"]["Get"][settings.weaviate_class_name]:
                    metadata = {
                        "content_type": item.get("content_type", ""),
                        "page_number": item.get("page_number", 0),
                        "chunk_index": item.get("chunk_index", 0),
                        "document_name": item.get("document_name", ""),
                        "bbox": item.get("bbox", ""),
                    }
                    
                    # Parse stored metadata
                    if item.get("metadata"):
                        try:
                            stored_metadata = json.loads(item["metadata"])
                            metadata.update(stored_metadata)
                        except json.JSONDecodeError:
                            pass
                    
                    doc = Document(
                        page_content=item.get("content", ""),
                        metadata=metadata
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    def delete_documents(self, document_name: str) -> bool:
        """Delete all chunks for a specific document."""
        try:
            where_clause = {
                "path": ["document_name"],
                "operator": "Equal",
                "valueString": document_name
            }
            
            result = self.client.batch.delete_objects(
                class_name=settings.weaviate_class_name,
                where=where_clause
            )
            
            print(f"Deleted documents for: {document_name}")
            return True
            
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            # Get total count
            result = (
                self.client.query
                .aggregate(settings.weaviate_class_name)
                .with_meta_count()
                .do()
            )
            
            total_count = 0
            if "data" in result and "Aggregate" in result["data"]:
                total_count = result["data"]["Aggregate"][settings.weaviate_class_name][0]["meta"]["count"]
            
            # Get breakdown by content type
            content_types = {}
            for content_type in ["text", "table", "figure"]:
                where_clause = {
                    "path": ["content_type"],
                    "operator": "Equal",
                    "valueString": content_type
                }
                
                result = (
                    self.client.query
                    .aggregate(settings.weaviate_class_name)
                    .with_where(where_clause)
                    .with_meta_count()
                    .do()
                )
                
                count = 0
                if "data" in result and "Aggregate" in result["data"]:
                    agg_result = result["data"]["Aggregate"][settings.weaviate_class_name]
                    if agg_result:
                        count = agg_result[0]["meta"]["count"]
                
                content_types[content_type] = count
            
            return {
                "total_chunks": total_count,
                "content_types": content_types
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_chunks": 0, "content_types": {}}