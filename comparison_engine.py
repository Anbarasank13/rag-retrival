"""
Document Comparison Engine
Compare multiple documents side-by-side
"""

from typing import List, Dict, Any
from collections import defaultdict
import difflib


class DocumentComparator:
    """
    Compare documents for similarities and differences
    """
    
    def compare_structure(
        self, 
        doc1_data: Dict[str, Any], 
        doc2_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare document structures
        
        Returns:
            Structural comparison results
        """
        
        doc1_structure = doc1_data.get('structure', {})
        doc2_structure = doc2_data.get('structure', {})
        
        doc1_sections = doc1_structure.get('sections', [])
        doc2_sections = doc2_structure.get('sections', [])
        
        comparison = {
            'doc1_name': doc1_data['name'],
            'doc2_name': doc2_data['name'],
            'doc1_sections': len(doc1_sections),
            'doc2_sections': len(doc2_sections),
            'common_section_titles': [],
            'unique_to_doc1': [],
            'unique_to_doc2': [],
            'hierarchy_comparison': {}
        }
        
        # Extract section titles
        doc1_titles = {s['title'].lower(): s for s in doc1_sections}
        doc2_titles = {s['title'].lower(): s for s in doc2_sections}
        
        # Find common sections
        common_titles = set(doc1_titles.keys()) & set(doc2_titles.keys())
        comparison['common_section_titles'] = list(common_titles)
        
        # Find unique sections
        unique_doc1 = set(doc1_titles.keys()) - set(doc2_titles.keys())
        unique_doc2 = set(doc2_titles.keys()) - set(doc1_titles.keys())
        
        comparison['unique_to_doc1'] = [doc1_titles[t]['title'] for t in unique_doc1]
        comparison['unique_to_doc2'] = [doc2_titles[t]['title'] for t in unique_doc2]
        
        # Compare hierarchy levels
        doc1_levels = defaultdict(int)
        doc2_levels = defaultdict(int)
        
        for section in doc1_sections:
            doc1_levels[section['level']] += 1
        
        for section in doc2_sections:
            doc2_levels[section['level']] += 1
        
        comparison['hierarchy_comparison'] = {
            'doc1_levels': dict(doc1_levels),
            'doc2_levels': dict(doc2_levels)
        }
        
        return comparison
    
    def compare_clauses(
        self,
        doc1_data: Dict[str, Any],
        doc2_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Compare clauses between documents
        
        Returns:
            List of clause comparisons
        """
        from clause_extractor import ClauseExtractor
        
        extractor = ClauseExtractor()
        
        # Extract clauses from both documents
        doc1_clauses = extractor.extract_clauses(
            {doc1_data['name']: doc1_data}
        )
        doc2_clauses = extractor.extract_clauses(
            {doc2_data['name']: doc2_data}
        )
        
        comparisons = []
        
        # Compare each clause type
        for clause_type in extractor.CLAUSE_TYPES.keys():
            clauses1 = doc1_clauses.get(clause_type, [])
            clauses2 = doc2_clauses.get(clause_type, [])
            
            if clauses1 or clauses2:
                comparison = {
                    'clause_type': clause_type,
                    'doc1_name': doc1_data['name'],
                    'doc2_name': doc2_data['name'],
                    'doc1_count': len(clauses1),
                    'doc2_count': len(clauses2),
                    'doc1_summary': self._summarize_clauses(clauses1),
                    'doc2_summary': self._summarize_clauses(clauses2),
                }
                
                comparisons.append(comparison)
        
        return comparisons
    
    def compare_entities(
        self,
        doc1_data: Dict[str, Any],
        doc2_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Compare entities extracted from documents
        """
        from knowledge_graph import KnowledgeGraphBuilder
        
        kg_builder = KnowledgeGraphBuilder(use_spacy=False)
        
        # Extract entities
        doc1_entities = kg_builder.extract_entities(doc1_data)
        doc2_entities = kg_builder.extract_entities(doc2_data)
        
        # Group by type
        doc1_by_type = defaultdict(list)
        doc2_by_type = defaultdict(list)
        
        for entity in doc1_entities:
            doc1_by_type[entity['type']].append(entity['text'])
        
        for entity in doc2_entities:
            doc2_by_type[entity['type']].append(entity['text'])
        
        return {
            doc1_data['name']: dict(doc1_by_type),
            doc2_data['name']: dict(doc2_by_type)
        }
    
    def compare_content_similarity(
        self,
        doc1_data: Dict[str, Any],
        doc2_data: Dict[str, Any],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Compare content similarity between documents
        
        Uses difflib to find similar chunks
        """
        
        doc1_chunks = [chunk.page_content for chunk in doc1_data['chunks']]
        doc2_chunks = [chunk.page_content for chunk in doc2_data['chunks']]
        
        similar_chunks = []
        
        for i, chunk1 in enumerate(doc1_chunks):
            for j, chunk2 in enumerate(doc2_chunks):
                # Calculate similarity ratio
                similarity = difflib.SequenceMatcher(
                    None, 
                    chunk1.lower(), 
                    chunk2.lower()
                ).ratio()
                
                if similarity >= threshold:
                    similar_chunks.append({
                        'doc1_chunk_index': i,
                        'doc2_chunk_index': j,
                        'similarity': similarity,
                        'doc1_section': doc1_data['chunks'][i].metadata.get('section', 'N/A'),
                        'doc2_section': doc2_data['chunks'][j].metadata.get('section', 'N/A'),
                        'doc1_preview': chunk1[:100] + '...',
                        'doc2_preview': chunk2[:100] + '...',
                    })
        
        # Sort by similarity
        similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'doc1_name': doc1_data['name'],
            'doc2_name': doc2_data['name'],
            'total_doc1_chunks': len(doc1_chunks),
            'total_doc2_chunks': len(doc2_chunks),
            'similar_chunks_count': len(similar_chunks),
            'similarity_threshold': threshold,
            'similar_chunks': similar_chunks[:10]  # Top 10
        }
    
    def compare_metadata(
        self,
        doc1_data: Dict[str, Any],
        doc2_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare document metadata
        """
        
        doc1_meta = doc1_data.get('metadata', {})
        doc2_meta = doc2_data.get('metadata', {})
        
        comparison = {
            'doc1_name': doc1_data['name'],
            'doc2_name': doc2_data['name'],
            'metadata_comparison': {}
        }
        
        # Compare common metadata fields
        common_fields = set(doc1_meta.keys()) & set(doc2_meta.keys())
        
        for field in common_fields:
            comparison['metadata_comparison'][field] = {
                'doc1': doc1_meta[field],
                'doc2': doc2_meta[field],
                'same': doc1_meta[field] == doc2_meta[field]
            }
        
        # Fields unique to each document
        comparison['unique_to_doc1'] = list(set(doc1_meta.keys()) - set(doc2_meta.keys()))
        comparison['unique_to_doc2'] = list(set(doc2_meta.keys()) - set(doc1_meta.keys()))
        
        return comparison
    
    def generate_comparison_report(
        self,
        doc1_data: Dict[str, Any],
        doc2_data: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive comparison report
        """
        
        report_sections = []
        
        # Header
        report_sections.append("=" * 80)
        report_sections.append("DOCUMENT COMPARISON REPORT")
        report_sections.append("=" * 80)
        report_sections.append(f"\nDocument 1: {doc1_data['name']}")
        report_sections.append(f"Document 2: {doc2_data['name']}\n")
        
        # Structure comparison
        report_sections.append("\n" + "-" * 80)
        report_sections.append("1. STRUCTURE COMPARISON")
        report_sections.append("-" * 80)
        
        structure_comp = self.compare_structure(doc1_data, doc2_data)
        report_sections.append(f"Document 1 Sections: {structure_comp['doc1_sections']}")
        report_sections.append(f"Document 2 Sections: {structure_comp['doc2_sections']}")
        report_sections.append(f"Common Sections: {len(structure_comp['common_section_titles'])}")
        
        if structure_comp['common_section_titles']:
            report_sections.append("\nCommon Section Titles:")
            for title in structure_comp['common_section_titles'][:5]:
                report_sections.append(f"  • {title}")
        
        if structure_comp['unique_to_doc1']:
            report_sections.append(f"\nUnique to {doc1_data['name']}:")
            for title in structure_comp['unique_to_doc1'][:5]:
                report_sections.append(f"  • {title}")
        
        if structure_comp['unique_to_doc2']:
            report_sections.append(f"\nUnique to {doc2_data['name']}:")
            for title in structure_comp['unique_to_doc2'][:5]:
                report_sections.append(f"  • {title}")
        
        # Entity comparison
        report_sections.append("\n" + "-" * 80)
        report_sections.append("2. ENTITY COMPARISON")
        report_sections.append("-" * 80)
        
        entity_comp = self.compare_entities(doc1_data, doc2_data)
        
        for doc_name, entities_by_type in entity_comp.items():
            report_sections.append(f"\n{doc_name}:")
            for entity_type, entities in entities_by_type.items():
                report_sections.append(f"  {entity_type}: {len(entities)} found")
                if entities:
                    report_sections.append(f"    Examples: {', '.join(entities[:3])}")
        
        # Content similarity
        report_sections.append("\n" + "-" * 80)
        report_sections.append("3. CONTENT SIMILARITY")
        report_sections.append("-" * 80)
        
        similarity_comp = self.compare_content_similarity(doc1_data, doc2_data)
        report_sections.append(f"Similar Chunks Found: {similarity_comp['similar_chunks_count']}")
        report_sections.append(f"Similarity Threshold: {similarity_comp['similarity_threshold']}")
        
        if similarity_comp['similar_chunks']:
            report_sections.append("\nTop Similar Sections:")
            for chunk in similarity_comp['similar_chunks'][:3]:
                report_sections.append(f"\n  Similarity: {chunk['similarity']:.2%}")
                report_sections.append(f"  Doc1 Section: {chunk['doc1_section']}")
                report_sections.append(f"  Doc2 Section: {chunk['doc2_section']}")
        
        # Summary
        report_sections.append("\n" + "=" * 80)
        report_sections.append("SUMMARY")
        report_sections.append("=" * 80)
        
        total_chunks1 = len(doc1_data['chunks'])
        total_chunks2 = len(doc2_data['chunks'])
        
        report_sections.append(f"Document 1: {total_chunks1} chunks, {structure_comp['doc1_sections']} sections")
        report_sections.append(f"Document 2: {total_chunks2} chunks, {structure_comp['doc2_sections']} sections")
        report_sections.append(f"Overall Similarity: {(similarity_comp['similar_chunks_count'] / max(total_chunks1, total_chunks2)) * 100:.1f}%")
        
        return "\n".join(report_sections)
    
    def _summarize_clauses(self, clauses: List[Dict[str, Any]]) -> str:
        """Generate summary of clauses"""
        
        if not clauses:
            return "No clauses found"
        
        # Get key information from first clause
        clause = clauses[0]
        
        key_terms = self._extract_clause_key_terms(clause)
        
        summary_parts = []
        
        if key_terms.get('amounts'):
            summary_parts.append(f"Amount: {key_terms['amounts'][0]}")
        
        if key_terms.get('durations'):
            summary_parts.append(f"Duration: {key_terms['durations'][0]}")
        
        if key_terms.get('dates'):
            summary_parts.append(f"Date: {key_terms['dates'][0]}")
        
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return f"Section: {clause['section']}"
    
    def _extract_clause_key_terms(self, clause: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract key terms from clause for summary"""
        import re
        
        content = clause['content']
        terms = {}
        
        # Extract amounts
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', content)
        terms['amounts'] = amounts[:2]
        
        # Extract durations
        durations = re.findall(r'(\d+)\s+(days?|weeks?|months?|years?)', content, re.IGNORECASE)
        terms['durations'] = [f"{num} {unit}" for num, unit in durations[:2]]
        
        # Extract dates
        dates = re.findall(
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
            content,
            re.IGNORECASE
        )
        terms['dates'] = dates[:2]
        
        return terms


# Example usage
def demo_comparison():
    """Demonstrate document comparison"""
    from langchain.schema import Document
    
    # Sample documents
    doc1 = {
        'name': 'contract_A.pdf',
        'chunks': [
            Document(
                page_content="This agreement is effective January 1, 2024. Party A agrees to pay Party B $100,000.",
                metadata={'section': '1. Payment Terms', 'page': 1}
            ),
            Document(
                page_content="Either party may terminate with 30 days notice.",
                metadata={'section': '5. Termination', 'page': 3}
            ),
        ],
        'structure': {
            'sections': [
                {'id': '1', 'title': 'Payment Terms', 'level': 1},
                {'id': '5', 'title': 'Termination', 'level': 1},
            ]
        },
        'metadata': {
            'total_pages': 3,
            'word_count': 500,
            'parties': ['Party A', 'Party B'],
        }
    }
    
    doc2 = {
        'name': 'contract_B.pdf',
        'chunks': [
            Document(
                page_content="This agreement commences on February 1, 2024. Party X will pay Party Y $75,000.",
                metadata={'section': '1. Payment Terms', 'page': 1}
            ),
            Document(
                page_content="Termination requires 60 days written notice.",
                metadata={'section': '6. Termination Clause', 'page': 4}
            ),
        ],
        'structure': {
            'sections': [
                {'id': '1', 'title': 'Payment Terms', 'level': 1},
                {'id': '6', 'title': 'Termination Clause', 'level': 1},
            ]
        },
        'metadata': {
            'total_pages': 4,
            'word_count': 650,
            'parties': ['Party X', 'Party Y'],
        }
    }
    
    # Compare documents
    comparator = DocumentComparator()
    
    print("Generating comparison report...\n")
    report = comparator.generate_comparison_report(doc1, doc2)
    print(report)


if __name__ == "__main__":
    demo_comparison()
