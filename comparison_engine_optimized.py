"""
Optimized Document Comparison Engine
Faster comparisons with better visualizations
"""

from typing import List, Dict, Any
from collections import defaultdict
import difflib


class DocumentComparator:
    """Optimized document comparison"""
    
    def compare_structure(
        self, 
        doc1_data: Dict[str, Any], 
        doc2_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare document structures"""
        
        doc1_sections = doc1_data.get('structure', {}).get('sections', [])
        doc2_sections = doc2_data.get('structure', {}).get('sections', [])
        
        # Extract titles
        doc1_titles = {s['title'].lower() for s in doc1_sections}
        doc2_titles = {s['title'].lower() for s in doc2_sections}
        
        # Find commonalities and differences
        common = doc1_titles & doc2_titles
        unique_doc1 = doc1_titles - doc2_titles
        unique_doc2 = doc2_titles - doc1_titles
        
        return {
            'doc1_name': doc1_data['name'],
            'doc2_name': doc2_data['name'],
            'doc1_sections': len(doc1_sections),
            'doc2_sections': len(doc2_sections),
            'common_section_titles': list(common),
            'unique_to_doc1': list(unique_doc1),
            'unique_to_doc2': list(unique_doc2),
        }
    
    def compare_clauses(
        self,
        doc1_data: Dict[str, Any],
        doc2_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare clauses between documents"""
        from clause_extractor import ClauseExtractor
        
        extractor = ClauseExtractor()
        
        # Extract clauses
        doc1_clauses = extractor.extract_clauses({doc1_data['name']: doc1_data})
        doc2_clauses = extractor.extract_clauses({doc2_data['name']: doc2_data})
        
        comparisons = []
        
        # Compare each type
        all_types = set(doc1_clauses.keys()) | set(doc2_clauses.keys())
        
        for clause_type in all_types:
            clauses1 = doc1_clauses.get(clause_type, [])
            clauses2 = doc2_clauses.get(clause_type, [])
            
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
        """Compare extracted entities"""
        from knowledge_graph_optimized import KnowledgeGraphBuilder
        
        kg_builder = KnowledgeGraphBuilder()
        
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
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Compare content similarity - OPTIMIZED"""
        
        # Limit chunks for performance
        doc1_chunks = [c.page_content for c in doc1_data['chunks'][:20]]
        doc2_chunks = [c.page_content for c in doc2_data['chunks'][:20]]
        
        similar_chunks = []
        
        # Compare chunks
        for i, chunk1 in enumerate(doc1_chunks):
            for j, chunk2 in enumerate(doc2_chunks):
                # Quick similarity check
                similarity = difflib.SequenceMatcher(
                    None, 
                    chunk1.lower()[:500],  # Limit comparison length
                    chunk2.lower()[:500]
                ).ratio()
                
                if similarity >= threshold:
                    similar_chunks.append({
                        'doc1_chunk_index': i,
                        'doc2_chunk_index': j,
                        'similarity': similarity,
                        'doc1_section': doc1_data['chunks'][i].metadata.get('section', 'N/A'),
                        'doc2_section': doc2_data['chunks'][j].metadata.get('section', 'N/A'),
                    })
        
        # Sort by similarity
        similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'doc1_name': doc1_data['name'],
            'doc2_name': doc2_data['name'],
            'total_doc1_chunks': len(doc1_data['chunks']),
            'total_doc2_chunks': len(doc2_data['chunks']),
            'similar_chunks_count': len(similar_chunks),
            'similarity_threshold': threshold,
            'similar_chunks': similar_chunks[:10]
        }
    
    def _summarize_clauses(self, clauses: List[Dict[str, Any]]) -> str:
        """Generate clause summary"""
        if not clauses:
            return "None found"
        
        clause = clauses[0]
        section = clause.get('section', 'N/A')
        return f"Found in {section}"
