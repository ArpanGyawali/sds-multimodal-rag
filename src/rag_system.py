"""Main RAG system implementation."""

import json
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from config.settings import settings
from .pdf_processor import process_pdf_file
from .vector_store import WeaviateVectorStore


class RAGSystem:
    """Main RAG system for PDF document Q&A."""
    
    def __init__(self):
        self.vector_store = WeaviateVectorStore()
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.openai_api_key,
            openai_api_base="https://models.github.ai/inference"
        )
        self.parser = StrOutputParser()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{question}")
        ])
        
        # Create the chain
        self.chain = (
            RunnablePassthrough.assign(context=self._get_context)
            | self.prompt
            | self.llm
            | self.parser
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are an expert document analyst with the ability to analyze text, tables, and figures from PDF documents. 
        
Your task is to answer questions based on the provided document context. The context may include:
- Text content from the document
- Tables with structured data
- Figures/images with descriptions

When answering:
1. Base your response strictly on the provided context
2. If you reference specific data, mention the page number when available
3. For tables, present the data clearly and mention it's from a table
4. For figures, describe what you can infer from the description provided
5. If the question cannot be answered from the context, say so clearly
6. Be precise and cite specific numbers, dates, or facts when available

Context: {context}

Remember to be accurate and cite page numbers when referencing specific information."""
    
    @traceable
    def _get_context(self, inputs: Dict[str, Any]) -> str:
        """Retrieve relevant context for the query."""
        question = inputs["question"]
        
        # Perform similarity search
        docs = self.vector_store.similarity_search(
            query=question,
            k=10  # Get more documents for better context
        )
        
        if not docs:
            return "No relevant context found."
        
        # Format context by content type
        context_parts = []
        
        for doc in docs:
            content_type = doc.metadata.get("content_type", "text")
            page_num = doc.metadata.get("page_number", "unknown")
            
            if content_type == "table":
                context_parts.append(
                    f"[TABLE from page {page_num}]\n{doc.page_content}\n"
                )
            elif content_type == "figure":
                # For figures, we include the description and potentially the image
                image_data = doc.metadata.get("image_data")
                if image_data:
                    context_parts.append(
                        f"[FIGURE from page {page_num}]\n{doc.page_content}\n[Image data available for visual analysis]\n"
                    )
                else:
                    context_parts.append(
                        f"[FIGURE from page {page_num}]\n{doc.page_content}\n"
                    )
            else:  # text
                context_parts.append(
                    f"[TEXT from page {page_num}]\n{doc.page_content}\n"
                )
        
        return "\n---\n".join(context_parts)
    
    @traceable
    def ingest_document(self, pdf_path: str) -> Dict[str, Any]:
        """Ingest a PDF document into the vector store."""
        print(f"Processing document: {pdf_path}")
        
        # Process the PDF
        chunks = process_pdf_file(pdf_path)
        
        if not chunks:
            return {
                "success": False,
                "message": "No content extracted from the document",
                "chunks_added": 0
            }
        
        # Add to vector store
        try:
            object_ids = self.vector_store.add_documents(chunks)
            
            return {
                "success": True,
                "message": f"Successfully processed {len(chunks)} chunks",
                "chunks_added": len(chunks),
                "object_ids": object_ids,
                "document_stats": {
                    "text_chunks": len([c for c in chunks if c.content_type == "text"]),
                    "table_chunks": len([c for c in chunks if c.content_type == "table"]),
                    "figure_chunks": len([c for c in chunks if c.content_type == "figure"])
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error adding documents to vector store: {str(e)}",
                "chunks_added": 0
            }
    
    @traceable
    def query(self, question: str, filters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Query the RAG system."""
        try:
            # Apply filters if provided
            if filters:
                docs = self.vector_store.similarity_search(
                    query=question,
                    k=10,
                    content_type=filters.get("content_type"),
                    document_name=filters.get("document_name")
                )
            else:
                docs = self.vector_store.similarity_search(question, k=10)
            
            # Generate response using the chain
            response = self.chain.invoke({"question": question})
            
            # Extract sources
            sources = []
            for doc in docs:
                sources.append({
                    "page_number": doc.metadata.get("page_number"),
                    "content_type": doc.metadata.get("content_type"),
                    "document_name": doc.metadata.get("document_name"),
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            return {
                "success": True,
                "answer": response,
                "sources": sources,
                "question": question
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": "I apologize, but I encountered an error while processing your question.",
                "sources": [],
                "question": question
            }
    
    def setup_vector_store(self, reset=False):
        """Setup vector store with optional reset"""
        print("🔧 Setting up vector store...")
        
        if not self.vector_store.test_connection():
            raise Exception("Cannot connect to Weaviate")
        
        # Pass reset parameter to schema creation
        if not self.vector_store.create_schema(reset=reset):
            raise Exception("Failed to create or verify schema")
        
        print("✅ Vector store setup complete")
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.vector_store.get_stats()
    
    def delete_document(self, document_name: str) -> bool:
        """Delete a document from the vector store."""
        return self.vector_store.delete_documents(document_name)


class MultimodalRAGSystem(RAGSystem):
    """Enhanced RAG system with multimodal capabilities for handling images."""
    
    def __init__(self):
        super().__init__()
        # Use GPT-4V for multimodal capabilities
        self.multimodal_llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.openai_api_key
        )
    
    @traceable
    def _analyze_image_content(self, image_data: str, question: str) -> str:
        """Analyze image content using GPT-4V."""
        try:
            messages = [
                SystemMessage(content="""You are an expert at analyzing images from documents. 
                Describe what you see in the image in detail, focusing on any text, charts, graphs, 
                diagrams, or other relevant information that might help answer questions about the document."""),
                HumanMessage(content=[
                    {"type": "text", "text": f"Analyze this image in the context of this question: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ])
            ]
            
            response = self.multimodal_llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    @traceable
    def _get_context(self, inputs: Dict[str, Any]) -> str:
        """Enhanced context retrieval with image analysis."""
        question = inputs["question"]
        
        # Perform similarity search
        docs = self.vector_store.similarity_search(
            query=question,
            k=10
        )
        
        if not docs:
            return "No relevant context found."
        
        # Format context by content type
        context_parts = []
        
        for doc in docs:
            content_type = doc.metadata.get("content_type", "text")
            page_num = doc.metadata.get("page_number", "unknown")
            
            if content_type == "table":
                context_parts.append(
                    f"[TABLE from page {page_num}]\n{doc.page_content}\n"
                )
            elif content_type == "figure":
                # For figures, analyze the image if available
                image_data = doc.metadata.get("image_data")
                if image_data:
                    # Analyze the image content
                    image_analysis = self._analyze_image_content(image_data, question)
                    context_parts.append(
                        f"[FIGURE from page {page_num}]\n{doc.page_content}\n"
                        f"Visual Analysis: {image_analysis}\n"
                    )
                else:
                    context_parts.append(
                        f"[FIGURE from page {page_num}]\n{doc.page_content}\n"
                    )
            else:  # text
                context_parts.append(
                    f"[TEXT from page {page_num}]\n{doc.page_content}\n"
                )
        
        return "\n---\n".join(context_parts)