"""
Optimized Hierarchical Document Processor
Better chunking strategy with improved efficiency
"""

import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from collections import defaultdict


class HierarchicalDocumentProcessor:
    """Process documents with optimized hierarchical structure extraction"""
    
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Optimized text splitter with better separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",  # Triple newline (major section break)
                "\n\n",    # Double newline (paragraph break)
                "\n",      # Single newline
                ". ",      # Sentence end
                "; ",      # Semicolon
                ", ",      # Comma
                " ",       # Space
                ""         # Character
            ],
            length_function=len,
        )
    
    def process_document(self, documents: List[Document], doc_name: str) -> Dict[str, Any]:
        """
        Process document with optimized chunking
        
        Returns:
            Dict with chunks, structure, and metadata
        """
        # Combine pages
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Clean text
        full_text = self._clean_text(full_text)
        
        # Extract structure
        structure = self.extract_structure(full_text)
        
        # Create smart chunks
        chunks = self.create_smart_chunks(documents, structure, doc_name)
        
        # Extract metadata
        metadata = self.extract_metadata(documents, full_text)
        
        return {
            'name': doc_name,
            'chunks': chunks,
            'structure': structure,
            'metadata': metadata,
            'raw_documents': documents
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text of common artifacts"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers at start of lines
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove common PDF artifacts
        text = re.sub(r'\f', '\n\n', text)  # Form feed to paragraph break
        return text.strip()
    
    def extract_structure(self, text: str) -> Dict[str, Any]:
        """
        Extract hierarchical structure - OPTIMIZED
        """
        structure = {
            'sections': [],
            'hierarchy': defaultdict(list),
            'toc': []  # Table of contents
        }
        
        # Optimized patterns - ordered by priority
        patterns = [
            # Legal document headers
            (r'^(ARTICLE|SECTION|CHAPTER)\s+([IVXLCDM]+|\d+)[:\.\s]+(.+?)$', 'article', 1),
            
            # Numbered sections with title
            (r'^(\d+\.(?:\d+\.)*)\s+([A-Z][^\n]{3,100})$', 'numbered', None),
            
            # Lettered subsections
            (r'^\s*\(([a-z]|[ivx]+)\)\s+(.+?)$', 'lettered', 3),
            
            # All caps headers (3+ words)
            (r'^([A-Z][A-Z\s]{10,})$', 'header', 2),
        ]
        
        lines = text.split('\n')
        section_id_counter = defaultdict(int)
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            if not line or len(line) < 3:
                continue
            
            for pattern, section_type, level in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                
                if match:
                    # Extract section info
                    if section_type == 'article':
                        section_id = match.group(2)
                        section_title = match.group(3).strip()
                        hierarchy_level = level
                    
                    elif section_type == 'numbered':
                        section_id = match.group(1)
                        section_title = match.group(2).strip()
                        # Calculate level from dots
                        hierarchy_level = len(section_id.split('.'))
                    
                    elif section_type == 'lettered':
                        section_id = match.group(1)
                        section_title = match.group(2).strip()[:100]  # Limit title length
                        hierarchy_level = level
                    
                    else:  # header
                        section_id_counter[section_type] += 1
                        section_id = f"H{section_id_counter[section_type]}"
                        section_title = line.strip()
                        hierarchy_level = level
                    
                    # Create section entry
                    section = {
                        'id': section_id,
                        'title': section_title,
                        'type': section_type,
                        'level': hierarchy_level,
                        'line_number': line_num,
                        'full_text': line
                    }
                    
                    structure['sections'].append(section)
                    structure['hierarchy'][hierarchy_level].append(section)
                    
                    # Add to TOC if major section
                    if hierarchy_level <= 2:
                        structure['toc'].append({
                            'id': section_id,
                            'title': section_title,
                            'level': hierarchy_level
                        })
                    
                    break  # Don't match multiple patterns
        
        return structure
    
    def create_smart_chunks(
        self, 
        documents: List[Document], 
        structure: Dict[str, Any],
        doc_name: str
    ) -> List[Document]:
        """
        Create chunks with smart boundaries - OPTIMIZED
        
        Strategy:
        1. Prefer breaking at section boundaries
        2. Keep related content together
        3. Add overlap for context
        """
        chunks = []
        
        # Build section map for quick lookup
        section_map = self._build_section_map(documents, structure)
        
        # Split with standard splitter first
        base_chunks = self.splitter.split_documents(documents)
        
        # Enhance chunks with metadata
        for idx, chunk in enumerate(base_chunks):
            # Find matching section
            section_info = self._find_chunk_section(
                chunk.page_content, 
                section_map,
                chunk.metadata.get('page', 0)
            )
            
            # Build enhanced metadata
            enhanced_metadata = {
                'source': doc_name,
                'page': chunk.metadata.get('page', 'N/A'),
                'chunk_index': idx,
                'total_chunks': len(base_chunks),
                'section': section_info.get('title', 'Unknown'),
                'section_id': section_info.get('id', 'N/A'),
                'hierarchy_level': section_info.get('level', 0),
                'section_type': section_info.get('type', 'N/A'),
                'char_count': len(chunk.page_content),
                'word_count': len(chunk.page_content.split())
            }
            
            # Create enhanced chunk
            enhanced_chunk = Document(
                page_content=chunk.page_content,
                metadata={**chunk.metadata, **enhanced_metadata}
            )
            
            chunks.append(enhanced_chunk)
        
        return chunks
    
    def _build_section_map(self, documents: List[Document], structure: Dict[str, Any]) -> Dict[int, List[Dict]]:
        """Build map of page numbers to sections"""
        section_map = defaultdict(list)
        
        # Map sections to approximate pages based on line numbers
        total_lines = sum(len(doc.page_content.split('\n')) for doc in documents)
        avg_lines_per_page = total_lines / len(documents) if documents else 100
        
        for section in structure['sections']:
            approx_page = int(section['line_number'] / avg_lines_per_page) + 1
            section_map[approx_page].append(section)
        
        return section_map
    
    def _find_chunk_section(
        self, 
        chunk_text: str, 
        section_map: Dict[int, List[Dict]],
        chunk_page: int
    ) -> Dict[str, Any]:
        """Find most relevant section for a chunk - OPTIMIZED"""
        
        # Check current page and adjacent pages
        for page_offset in [0, -1, 1]:
            page_num = chunk_page + page_offset
            
            if page_num in section_map:
                # Look for section titles/ids in chunk text
                for section in reversed(section_map[page_num]):  # Reverse to get most recent
                    # Check if section appears in chunk
                    if (section['id'] in chunk_text[:200] or 
                        section['title'].lower() in chunk_text[:300].lower()):
                        return section
        
        # Fallback: check all sections
        chunk_lower = chunk_text[:400].lower()
        for section in structure.get('sections', []):
            if section['title'].lower() in chunk_lower:
                return section
        
        return {}
    
    def extract_metadata(self, documents: List[Document], full_text: str) -> Dict[str, Any]:
        """
        Extract document-level metadata - OPTIMIZED
        """
        metadata = {
            'total_pages': len(documents),
            'total_length': len(full_text),
            'word_count': len(full_text.split()),
            'avg_words_per_page': len(full_text.split()) // max(len(documents), 1)
        }
        
        # Limit text for pattern matching (first 5000 chars)
        sample_text = full_text[:5000]
        
        # Extract dates (limit to 10)
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, sample_text, re.IGNORECASE)
        metadata['dates_found'] = list(set(dates))[:10]
        
        # Extract parties/organizations (limit to 10)
        party_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:LLC|Inc\.|Corp\.|Corporation|Company|Ltd\.|Limited)\b'
        parties = re.findall(party_pattern, sample_text)
        metadata['parties'] = list(set(parties))[:10]
        
        # Extract monetary amounts (limit to 20)
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        amounts = re.findall(money_pattern, sample_text)
        metadata['monetary_amounts'] = list(set(amounts))[:20]
        
        return metadata


# Helper function to display structure
def display_structure(structure: Dict[str, Any], max_sections: int = 20):
    """Display document structure in readable format"""
    
    print("Document Structure:")
    print("=" * 60)
    print(f"Total Sections: {len(structure['sections'])}")
    print(f"Hierarchy Levels: {len(structure['hierarchy'])}")
    print("\nTable of Contents:")
    print("-" * 60)
    
    for item in structure['toc'][:max_sections]:
        indent = "  " * (item['level'] - 1)
        print(f"{indent}{item['id']}. {item['title']}")
    
    if len(structure['toc']) > max_sections:
        print(f"\n... and {len(structure['toc']) - max_sections} more sections")


# Demo function
def demo():
    """Demonstrate optimized processing"""
    from langchain.schema import Document
    
    sample_text = """
SERVICE AGREEMENT

ARTICLE I - DEFINITIONS AND INTERPRETATION

1.1 Definitions
In this Agreement, unless the context requires otherwise:
(a) "Effective Date" means January 1, 2024
(b) "Services" means the services described in Schedule A
(c) "Fee" means the amount specified in Section 3.1

1.2 Interpretation
References to sections are references to sections of this Agreement.

ARTICLE II - SCOPE OF SERVICES

2.1 Service Description
The Provider shall deliver the Services in accordance with Schedule A.

2.2 Standards
All Services shall meet industry standards and best practices.

ARTICLE III - PAYMENT TERMS

3.1 Fees
The Client shall pay the Provider $100,000 as follows:
(a) $30,000 upon execution
(b) $40,000 upon Phase 1 completion
(c) $30,000 upon final delivery

3.2 Payment Schedule
All payments are due within 30 days of invoice.

3.3 Late Payments
Late payments shall incur interest at 5% per annum.
"""
    
    doc = Document(page_content=sample_text, metadata={'page': 1})
    
    processor = HierarchicalDocumentProcessor()
    result = processor.process_document([doc], "sample_agreement.txt")
    
    print("Processing Results:")
    print("=" * 60)
    display_structure(result['structure'])
    
    print(f"\nChunks Created: {len(result['chunks'])}")
    print("\nSample Chunk:")
    print("-" * 60)
    print(f"Content: {result['chunks'][0].page_content[:200]}...")
    print(f"\nMetadata: {result['chunks'][0].metadata}")
    
    print(f"\nDocument Metadata:")
    print("-" * 60)
    for key, value in result['metadata'].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    demo()
