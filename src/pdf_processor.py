"""PDF processing module for extracting text, tables, and figures."""

import json
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import pymupdf4llm

from config.settings import settings


@dataclass
class DocumentChunk:
    """Represents a chunk of document content."""
    content: str
    content_type: str  # 'text', 'table', 'figure'
    page_number: int
    chunk_index: int
    document_name: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        if self.bbox:
            data['bbox'] = json.dumps(self.bbox)
        if self.metadata:
            data['metadata'] = json.dumps(self.metadata)
        else:
            data['metadata'] = json.dumps({})
        return data


class PDFProcessor:
    """Processes PDF documents to extract text, tables, and figures."""
    
    def __init__(self):
        self.doc = None
        self.document_name = ""
    
    def load_document(self, pdf_path: str) -> None:
        """Load a PDF document."""
        pdf_path = Path(pdf_path)
        self.document_name = pdf_path.stem
        self.doc = fitz.open(pdf_path)
    
    def extract_text_chunks(self, page_num: int) -> List[DocumentChunk]:
        """Extract text chunks from a page using pymupdf4llm for better structure."""
        if not self.doc:
            raise ValueError("No document loaded")
        
        # Use pymupdf4llm for better text extraction with structure
        page_text = pymupdf4llm.to_markdown(self.doc, pages=[page_num])
        
        if not page_text.strip():
            return []
        
        # Simple chunking - split by paragraphs and maintain size limits
        paragraphs = page_text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > settings.chunk_size and current_chunk:
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    content_type="text",
                    page_number=page_num + 1,  # 1-indexed
                    chunk_index=chunk_index,
                    document_name=self.document_name,
                    metadata={"extraction_method": "pymupdf4llm"}
                ))
                current_chunk = paragraph
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                content_type="text",
                page_number=page_num + 1,
                chunk_index=chunk_index,
                document_name=self.document_name,
                metadata={"extraction_method": "pymupdf4llm"}
            ))
        
        return chunks
    
    def extract_tables(self, page_num: int) -> List[DocumentChunk]:
        """Extract tables from a page."""
        if not self.doc:
            raise ValueError("No document loaded")
        
        page = self.doc[page_num]
        tables = page.find_tables()
        
        table_chunks = []
        for idx, table in enumerate(tables):
            try:
                # Extract table data
                table_data = table.extract()
                if not table_data:
                    continue
                
                # Convert to pandas DataFrame for better formatting
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                
                # Create markdown representation
                table_md = df.to_markdown(index=False)
                
                # Get bounding box
                bbox = table.bbox
                
                table_chunks.append(DocumentChunk(
                    content=f"Table {idx + 1}:\n{table_md}",
                    content_type="table",
                    page_number=page_num + 1,
                    chunk_index=idx,
                    document_name=self.document_name,
                    bbox=bbox,
                    metadata={
                        "table_index": idx,
                        "rows": len(table_data),
                        "columns": len(table_data[0]) if table_data else 0,
                        "extraction_method": "pymupdf"
                    }
                ))
            except Exception as e:
                print(f"Error extracting table {idx} from page {page_num + 1}: {e}")
                continue
        
        return table_chunks
    
    def extract_figures(self, page_num: int) -> List[DocumentChunk]:
        """Extract figures/images from a page."""
        if not self.doc:
            raise ValueError("No document loaded")
        
        page = self.doc[page_num]
        image_list = page.get_images()
        
        figure_chunks = []
        for idx, img in enumerate(image_list):
            try:
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(self.doc, xref)
                
                # Skip very small images (likely decorative)
                if pix.width < 100 or pix.height < 100:
                    pix = None
                    continue
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                pil_image = Image.open(BytesIO(img_data))
                
                # Convert to base64 for storage
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Get image location on page
                img_rects = page.get_image_rects(xref)
                bbox = img_rects[0] if img_rects else None
                
                figure_chunks.append(DocumentChunk(
                    content=f"Figure {idx + 1}: [Image data stored as base64]\nImage dimensions: {pix.width}x{pix.height}",
                    content_type="figure",
                    page_number=page_num + 1,
                    chunk_index=idx,
                    document_name=self.document_name,
                    bbox=bbox,
                    metadata={
                        "figure_index": idx,
                        "width": pix.width,
                        "height": pix.height,
                        "image_data": img_b64,
                        "extraction_method": "pymupdf"
                    }
                ))
                
                pix = None  # Clean up
                
            except Exception as e:
                print(f"Error extracting figure {idx} from page {page_num + 1}: {e}")
                continue
        
        return figure_chunks
    
    def process_document(self) -> List[DocumentChunk]:
        """Process entire document and extract all content."""
        if not self.doc:
            raise ValueError("No document loaded")
        
        all_chunks = []
        
        for page_num in range(len(self.doc)):
            print(f"Processing page {page_num + 1}/{len(self.doc)}")
            
            # Extract different types of content
            text_chunks = self.extract_text_chunks(page_num)
            table_chunks = self.extract_tables(page_num)
            figure_chunks = self.extract_figures(page_num)
            
            # Combine all chunks from this page
            page_chunks = text_chunks + table_chunks + figure_chunks
            all_chunks.extend(page_chunks)
        
        print(f"Extracted {len(all_chunks)} chunks total")
        print(f"- Text chunks: {len([c for c in all_chunks if c.content_type == 'text'])}")
        print(f"- Table chunks: {len([c for c in all_chunks if c.content_type == 'table'])}")
        print(f"- Figure chunks: {len([c for c in all_chunks if c.content_type == 'figure'])}")
        
        return all_chunks
    
    def close(self):
        """Close the document."""
        if self.doc:
            self.doc.close()
            self.doc = None


def process_pdf_file(pdf_path: str) -> List[DocumentChunk]:
    """Convenience function to process a PDF file."""
    processor = PDFProcessor()
    try:
        processor.load_document(pdf_path)
        chunks = processor.process_document()
        return chunks
    finally:
        processor.close()