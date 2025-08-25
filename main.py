"""Main application entry point for the RAG system."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from src.rag_system import RAGSystem, MultimodalRAGSystem


# Load environment variables
load_dotenv()


@click.group()
def cli():
    """RAG-based PDF Document Q&A System"""
    pass


@cli.command()
@click.option('--reset', is_flag=True, help='Reset the vector store schema')
def setup(reset: bool):
    """Setup the vector store and schema."""
    try:
        rag = RAGSystem()
        rag.setup_vector_store(reset=reset)
        click.echo("✅ Vector store setup completed successfully!")
    except Exception as e:
        click.echo(f"❌ Setup failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--multimodal', is_flag=True, help='Use multimodal capabilities for image analysis')
def ingest(pdf_path: str, multimodal: bool):
    """Ingest a PDF document into the vector store."""
    try:
        if multimodal:
            rag = MultimodalRAGSystem()
        else:
            rag = RAGSystem()
        
        result = rag.ingest_document(pdf_path)
        
        if result["success"]:
            click.echo(f"✅ {result['message']}")
            click.echo(f"📊 Document Statistics:")
            stats = result["document_stats"]
            click.echo(f"   - Text chunks: {stats['text_chunks']}")
            click.echo(f"   - Table chunks: {stats['table_chunks']}")
            click.echo(f"   - Figure chunks: {stats['figure_chunks']}")
        else:
            click.echo(f"❌ Ingestion failed: {result['message']}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error during ingestion: {e}")
        sys.exit(1)


@cli.command()
@click.argument('question', type=str)
@click.option('--content-type', type=click.Choice(['text', 'table', 'figure']), help='Filter by content type')
@click.option('--document', type=str, help='Filter by document name')
@click.option('--multimodal', is_flag=True, help='Use multimodal capabilities for image analysis')
def query(question: str, content_type: Optional[str], document: Optional[str], multimodal: bool):
    """Query the RAG system with a question."""
    try:
        if multimodal:
            rag = MultimodalRAGSystem()
        else:
            rag = RAGSystem()
        
        filters = {}
        if content_type:
            filters['content_type'] = content_type
        if document:
            filters['document_name'] = document
        
        result = rag.query(question, filters=filters if filters else None)
        
        if result["success"]:
            click.echo(f"\n🤔 Question: {result['question']}")
            click.echo(f"\n💡 Answer:\n{result['answer']}")
            
            if result['sources']:
                click.echo(f"\n📚 Sources:")
                for i, source in enumerate(result['sources'][:3]):  # Show top 3 sources
                    click.echo(f"   {i+1}. Page {source['page_number']} ({source['content_type']}):")
                    click.echo(f"      {source['preview']}")
        else:
            click.echo(f"❌ Query failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        click.echo(f"❌ Error during query: {e}")
        sys.exit(1)


@cli.command()
def interactive(multimodal: bool = False):
    """Start interactive Q&A session."""
    click.echo("🚀 Starting interactive RAG session...")
    click.echo("Type 'exit' or 'quit' to end the session.")
    click.echo("Type 'stats' to see vector store statistics.")
    click.echo("Type 'help' for available commands.\n")
    
    try:
        if multimodal:
            rag = MultimodalRAGSystem()
        else:
            rag = RAGSystem()
    except Exception as e:
        click.echo(f"❌ Failed to initialize RAG system: {e}")
        return
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit']:
                click.echo("👋 Goodbye!")
                break
            
            if question.lower() == 'stats':
                stats = rag.get_vector_store_stats()
                click.echo(f"\n📊 Vector Store Statistics:")
                click.echo(f"   Total chunks: {stats['total_chunks']}")
                click.echo(f"   Content types: {stats['content_types']}")
                continue
            
            if question.lower() == 'help':
                click.echo("\n📖 Available commands:")
                click.echo("   - Ask any question about your documents")
                click.echo("   - 'stats' - Show vector store statistics")
                click.echo("   - 'exit' or 'quit' - End session")
                continue
            
            if not question:
                continue
            
            result = rag.query(question)
            
            if result["success"]:
                click.echo(f"\n💡 Answer:\n{result['answer']}")
                
                if result['sources']:
                    click.echo(f"\n📚 Top sources:")
                    for i, source in enumerate(result['sources'][:2]):
                        click.echo(f"   • Page {source['page_number']} ({source['content_type']})")
            else:
                click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            click.echo("\n👋 Goodbye!")
            break
        except Exception as e:
            click.echo(f"❌ Error: {e}")


@cli.command()
def stats():
    """Show vector store statistics."""
    try:
        rag = RAGSystem()
        stats = rag.get_vector_store_stats()
        
        click.echo("📊 Vector Store Statistics:")
        click.echo(f"   Total chunks: {stats['total_chunks']}")
        click.echo(f"   Content breakdown:")
        for content_type, count in stats['content_types'].items():
            click.echo(f"     - {content_type}: {count}")
            
    except Exception as e:
        click.echo(f"❌ Error getting stats: {e}")


@cli.command()
@click.argument('document_name', type=str)
def delete(document_name: str):
    """Delete a document from the vector store."""
    try:
        rag = RAGSystem()
        success = rag.delete_document(document_name)
        
        if success:
            click.echo(f"✅ Deleted document: {document_name}")
        else:
            click.echo(f"❌ Failed to delete document: {document_name}")
            
    except Exception as e:
        click.echo(f"❌ Error deleting document: {e}")


if __name__ == "__main__":
    cli()