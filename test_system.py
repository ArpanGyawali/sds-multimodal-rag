"""Test script for the RAG system."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from src.rag_system import RAGSystem, MultimodalRAGSystem


def test_basic_functionality():
    """Test basic RAG system functionality."""
    print("🧪 Testing RAG System...")
    
    # Initialize system
    rag = RAGSystem()
    
    # Test vector store setup
    print("Setting up vector store...")
    rag.setup_vector_store(reset=True)
    
    # Test stats (should be empty initially)
    stats = rag.get_vector_store_stats()
    print(f"Initial stats: {stats}")
    
    print("✅ Basic functionality test passed!")


def test_with_sample_document():
    """Test with a sample document if available."""
    print("🧪 Testing with document...")
    
    # Look for any PDF in the current directory
    pdf_files = list(Path(".").glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in current directory. Skipping document test.")
        return
    
    pdf_path = str(pdf_files[0])
    print(f"Using document: {pdf_path}")
    
    # Initialize system
    rag = MultimodalRAGSystem()
    
    # Setup
    rag.setup_vector_store(reset=True)
    
    # Ingest document
    print("Ingesting document...")
    result = rag.ingest_document(pdf_path)
    
    if result["success"]:
        print(f"✅ Ingestion successful: {result['message']}")
        print(f"Stats: {result['document_stats']}")
        
        # Test query
        print("Testing query...")
        query_result = rag.query("What is this document about?")
        
        if query_result["success"]:
            print(f"✅ Query successful!")
            print(f"Answer: {query_result['answer'][:200]}...")
            print(f"Sources: {len(query_result['sources'])} found")
        else:
            print(f"❌ Query failed: {query_result.get('error')}")
    else:
        print(f"❌ Ingestion failed: {result['message']}")


def test_vector_store_connection():
    """Test Weaviate connection."""
    print("🧪 Testing Weaviate connection...")
    
    try:
        from vector_store import WeaviateVectorStore
        
        vs = WeaviateVectorStore()
        print("✅ Weaviate connection successful!")
        
        # Test schema operations
        vs.create_schema()
        print("✅ Schema creation successful!")
        
        stats = vs.get_stats()
        print(f"Vector store stats: {stats}")
        
    except Exception as e:
        print(f"❌ Weaviate connection failed: {e}")
        print("Please check your Weaviate configuration in the .env file")


def run_all_tests():
    """Run all tests."""
    print("🚀 Starting RAG System Tests\n")
    
    # Check environment variables
    required_env_vars = [
        "OPENAI_API_KEY",
        "WEAVIATE_URL", 
        "WEAVIATE_API_KEY",
        "LANGCHAIN_API_KEY",
        "GOOGLE_API_KEY"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please check your .env file configuration.")
        return
    
    print("✅ Environment variables configured\n")
    
    # Run tests
    test_vector_store_connection()
    print()
    
    test_basic_functionality()
    print()
    
    test_with_sample_document()
    print()
    
    print("🎉 All tests completed!")


if __name__ == "__main__":
    run_all_tests()