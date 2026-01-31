"""
Hierarchical Document Processor
Extracts document structure, sections, and hierarchical information
"""

import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from collections import defaultdict


class HierarchicalDocumentProcessor:
    """Process documents with hierarchical structure extraction"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
    
    def process_document(self, documents: List[Document], doc_name: str) -> Dict[str, Any]:
        """
        Process a document and extract hierarchical structure
        
        Args:
            documents: List of LangChain Document objects
            doc_name: Name of the document
            
        Returns:
            Dictionary containing processed document data
        """
        # Combine all pages
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Extract structure
        structure = self.extract_structure(full_text)
        
        # Create chunks with enhanced metadata
        chunks = self.create_hierarchical_chunks(documents, structure, doc_name)
        
        # Extract metadata
        metadata = self.extract_metadata(documents, full_text)
        
        return {
            'name': doc_name,
            'chunks': chunks,
            'structure': structure,
            'metadata': metadata,
            'raw_documents': documents
        }
    
    def extract_structure(self, text: str) -> Dict[str, Any]:
        """
        Extract hierarchical structure from text
        
        Identifies:
        - Sections (H1, H2, H3 style headers)
        - Numbered sections (1., 1.1, 1.1.1)
        - Articles/Clauses
        """
        structure = {
            'sections': [],
            'hierarchy': defaultdict(list)
        }
        
        # Pattern for different header styles
        patterns = [
            # Article/Section style: "ARTICLE I", "SECTION 1"
            (r'^(ARTICLE|SECTION|CHAPTER)\s+([IVXLCDM]+|\d+)\s*[:\-]?\s*(.+)$', 'article'),
            
            # Numbered sections: "1.", "1.1", "1.1.1"
            (r'^(\d+(?:\.\d+)*)\s+(.+)$', 'numbered'),
            
            # Lettered sections: "(a)", "(i)", "a."
            (r'^(\([a-z]+\)|\([ivx]+\)|[a-z]\.)\s+(.+)$', 'lettered'),
            
            # ALL CAPS headers
            (r'^([A-Z][A-Z\s]{3,})$', 'header'),
        ]
        
        lines = text.split('\n')
        current_section = None
        section_number = 0
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            for pattern, section_type in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                
                if match:
                    section_number += 1
                    
                    if section_type == 'article':
                        section_id = match.group(2)
                        section_title = match.group(3).strip()
                        level = 1
                    elif section_type == 'numbered':
                        section_id = match.group(1)
                        section_title = match.group(2).strip()
                        level = len(section_id.split('.'))
                    elif section_type == 'lettered':
                        section_id = match.group(1)
                        section_title = match.group(2).strip()
                        level = 3
                    else:  # header
                        section_id = f"H{section_number}"
                        section_title = line
                        level = 2
                    
                    section = {
                        'id': section_id,
                        'title': section_title,
                        'type': section_type,
                        'level': level,
                        'line_number': line_num,
                        'full_text': line
                    }
                    
                    structure['sections'].append(section)
                    structure['hierarchy'][level].append(section)
                    current_section = section
                    break
        
        return structure
    
    def create_hierarchical_chunks(
        self, 
        documents: List[Document], 
        structure: Dict[str, Any],
        doc_name: str
    ) -> List[Document]:
        """
        Create chunks with hierarchical metadata
        """
        chunks = []
        
        # Split documents into chunks
        all_chunks = self.splitter.split_documents(documents)
        
        # Enhance each chunk with hierarchical metadata
        for chunk in all_chunks:
            # Find the section this chunk belongs to
            chunk_text = chunk.page_content
            matching_section = self.find_matching_section(chunk_text, structure)
            
            # Enhanced metadata
            enhanced_metadata = {
                'source': doc_name,
                'page': chunk.metadata.get('page', 'N/A'),
                'chunk_index': len(chunks),
                'section': matching_section.get('title', 'Unknown') if matching_section else 'Unknown',
                'section_id': matching_section.get('id', 'N/A') if matching_section else 'N/A',
                'hierarchy_level': matching_section.get('level', 0) if matching_section else 0,
                'section_type': matching_section.get('type', 'N/A') if matching_section else 'N/A',
            }
            
            # Create new document with enhanced metadata
            enhanced_chunk = Document(
                page_content=chunk.page_content,
                metadata={**chunk.metadata, **enhanced_metadata}
            )
            
            chunks.append(enhanced_chunk)
        
        return chunks
    
    def find_matching_section(self, chunk_text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find which section a chunk belongs to based on text matching
        """
        # Get first line of chunk
        first_lines = chunk_text.split('\n')[:3]
        
        # Try to find matching section
        for section in structure['sections']:
            # Check if section title/id appears in chunk
            if section['title'].lower() in chunk_text.lower()[:200]:
                return section
            if section['id'] in chunk_text[:100]:
                return section
        
        return {}
    
    def extract_metadata(self, documents: List[Document], full_text: str) -> Dict[str, Any]:
        """
        Extract document-level metadata
        """
        metadata = {
            'total_pages': len(documents),
            'total_length': len(full_text),
            'word_count': len(full_text.split()),
        }
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, full_text, re.IGNORECASE)
        metadata['dates_found'] = list(set(dates))[:10]  # First 10 unique dates
        
        # Extract potential parties (capitalized names followed by common legal terms)
        party_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:LLC|Inc\.|Corp\.|Corporation|Company|Ltd\.|Limited)\b'
        parties = re.findall(party_pattern, full_text)
        metadata['parties'] = list(set(parties))[:10]
        
        # Extract monetary amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        amounts = re.findall(money_pattern, full_text)
        metadata['monetary_amounts'] = list(set(amounts))[:20]
        
        return metadata


class SectionNavigator:
    """Navigate document sections hierarchically"""
    
    def __init__(self, structure: Dict[str, Any]):
        self.structure = structure
        self.build_tree()
    
    def build_tree(self):
        """Build tree structure from flat section list"""
        self.tree = {}
        
        for section in self.structure['sections']:
            level = section['level']
            section_id = section['id']
            
            if level == 1:
                self.tree[section_id] = {
                    'data': section,
                    'children': {}
                }
    
    def get_section(self, section_id: str) -> Dict[str, Any]:
        """Get section by ID"""
        for section in self.structure['sections']:
            if section['id'] == section_id:
                return section
        return {}
    
    def get_subsections(self, section_id: str) -> List[Dict[str, Any]]:
        """Get all subsections under a section"""
        subsections = []
        parent_section = self.get_section(section_id)
        
        if not parent_section:
            return subsections
        
        parent_level = parent_section['level']
        parent_index = self.structure['sections'].index(parent_section)
        
        # Find all sections that come after this one with higher level
        for section in self.structure['sections'][parent_index + 1:]:
            if section['level'] <= parent_level:
                break
            if section['level'] == parent_level + 1:
                subsections.append(section)
        
        return subsections
    
    def get_parent_section(self, section_id: str) -> Dict[str, Any]:
        """Get parent section"""
        current = self.get_section(section_id)
        
        if not current:
            return {}
        
        current_level = current['level']
        current_index = self.structure['sections'].index(current)
        
        # Search backwards for section with level - 1
        for section in reversed(self.structure['sections'][:current_index]):
            if section['level'] == current_level - 1:
                return section
        
        return {}
    
    def get_section_path(self, section_id: str) -> List[str]:
        """Get full path to section (breadcrumb)"""
        path = []
        current = self.get_section(section_id)
        
        while current:
            path.insert(0, current['title'])
            current = self.get_parent_section(current['id'])
        
        return path


# Example usage functions
def demo_hierarchical_processing():
    """Demonstrate hierarchical processing"""
    
    sample_text = """
ARTICLE I - DEFINITIONS

1.1 General Definitions
This Agreement defines the following terms...

1.2 Specific Terms
(a) "Effective Date" means January 1, 2024
(b) "Party A" refers to ABC Corporation
(c) "Party B" refers to XYZ Limited

ARTICLE II - PAYMENT TERMS

2.1 Payment Schedule
Party A shall pay Party B the sum of $50,000.00

2.2 Late Fees
(a) Late payments incur 5% penalty
(b) After 30 days, additional 2% per month

ARTICLE III - TERMINATION

3.1 Termination for Cause
Either party may terminate with 30 days notice...
"""
    
    # Create mock document
    from langchain.schema import Document
    doc = Document(page_content=sample_text, metadata={'page': 1})
    
    processor = HierarchicalDocumentProcessor()
    result = processor.process_document([doc], "sample_contract.txt")
    
    print("Structure extracted:")
    print(f"Total sections: {len(result['structure']['sections'])}")
    
    for section in result['structure']['sections']:
        indent = "  " * (section['level'] - 1)
        print(f"{indent}{section['id']}: {section['title']} (Level {section['level']})")
    
    print(f"\nTotal chunks created: {len(result['chunks'])}")
    print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    demo_hierarchical_processing()
