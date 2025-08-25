# RAG-Based PDF Document Q&A System

A comprehensive Retrieval-Augmented Generation (RAG) system designed to process and query PDF documents containing text, tables, and figures. Built with Weaviate cloud, LangChain, and multimodal LLM capabilities.

## Features

- **Multi-Content Processing**: Handles text, tables, and figures from PDF documents
- **Multimodal Analysis**: Used GPT-4o Vision for image and chart analysis (you can change in .env file)
- **Intelligent Chunking**: Preserves document structure and context
- **Advanced Search**: Vector similarity search with metadata filtering
- **Full Observability**: LangSmith integration for tracing and evaluation
- **CLI Interface**: User-friendly command-line interface
- **Interactive Mode**: Conversational document analysis

## Architecture

![Architecture](asset/architecture.png)

## Requirements

- Python 3.8+
- OpenAI API key   (Used openai for multimodel, can change this)
- GenAI API key   (Used gemini for embedding model, can change this)
- Weaviate Cloud account    (Can you weaviate locally through docker)
- LangSmith account (optional)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/ArpanGyawali/sds-multimodal-rag.git
cd sds-multimodal-rag
```

### 2. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Your .env fils should look like this

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o   #or any other model

# GenAI configuration
GOOGLE_API_KEY=AIzaSyDAApJe5GDOxHC2yfuis9OhqaoOqYwi6y4
EMBEDDING_MODEL=models/text-embedding-004             #gemini free embedding model

# Weaviate Cloud
WEAVIATE_URL=https://your-cluster-url.weaviate.network
WEAVIATE_API_KEY=your_weaviate_api_key
WEAVIATE_CLASS_NAME=your_weaviate_class_name

# LangSmith (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key

# Processing Settings (Change by your requirement)
CHUNK_SIZE=512
CHUNK_OVERLAP=10
MAX_TOKENS=4000
TEMPERATURE=0.1

```

### 4. Initialize System
```bash
python main.py setup
```

## Quick Start

### Test the system
```bash
# The document should be in the same folder as test_system.py for this test
python test_system.py
```
This will:
- Test Weaviate connection
- Verify basic functionality
- Process sample documents (if available)
- Validate query operations

## Usage Examples

### Financial Document Analysis
```bash
# Process Apple's Q3 2022 financial report.Put correct path. Remove --multimodal if you want text only modal
python main.py ingest "2022_Q3_AAPL.pdf" --multimodal    

# Query revenue information
python main.py query "What was Apple's total revenue in Q3 2022?"

# Analyze tables specifically
python main.py query "Show me the balance sheet data" --content-type table

# Understand charts and figures if necessary. 
python main.py query "What do the financial charts reveal?" --multimodal

# USEFUL: Interactive mode, where you can query and get answer back and forth through cli in multimodal mode.
python main.py interactive
```


## System Capabilities

### Content Type Support
- **Text**: Paragraphs, sections, and structured content
- **Tables**: Financial data, statistics, structured information
- **Figures**: Charts, graphs, diagrams, and images

### Advanced Features
- **Semantic Search**: Vector-based similarity matching
- **Metadata Filtering**: Filter by content type, document, or page
- **Multimodal Analysis**: Visual understanding of charts and images
- **Context Preservation**: Maintains document structure and citations
- **Batch Processing**: Efficient handling of large documents

## Available Commands

```bash
# Setup and management
python main.py setup [--reset]           # Initialize vector store
python main.py stats                     # Show system statistics
python main.py delete <document-name>    # Remove document

# Document processing
python main.py ingest <pdf-path> [--multimodal]

# Querying
python main.py query "<question>" [--content-type TYPE] [--document DOC] [--multimodal]
python main.py interactive              # Start interactive session

# Testing
python test_system.py                   # Run system tests
```

## Project Structure

```
pdf-rag-system/
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── vector_store.py
│   └── rag_system.py
├── main.py
├── test_system.py
├── requirements.txt
├── .env
└── README.md
```



