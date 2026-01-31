"""
Optimized Knowledge Graph Builder
Faster extraction with cleaner graphs
"""

import re
from typing import List, Dict, Any
import networkx as nx
from collections import defaultdict


class KnowledgeGraphBuilder:
    """Build optimized knowledge graphs"""
    
    def __init__(self):
        self.max_entities_per_doc = 50  # Limit for clarity
        self.max_relationships = 100
    
    def build_from_documents(self, documents: Dict[str, Any]) -> nx.DiGraph:
        """Build graph with entity limiting"""
        graph = nx.DiGraph()
        
        all_entities = []
        all_relationships = []
        
        for doc_name, doc_data in documents.items():
            entities = self.extract_entities(doc_data)
            relationships = self.extract_relationships(doc_data, entities)
            
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        # Limit entities
        if len(all_entities) > self.max_entities_per_doc:
            # Keep most important (by frequency)
            entity_counts = defaultdict(int)
            for entity in all_entities:
                entity_counts[entity['text']] += 1
            
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
            top_entity_texts = {e[0] for e in top_entities[:self.max_entities_per_doc]}
            
            all_entities = [e for e in all_entities if e['text'] in top_entity_texts]
        
        # Add to graph
        self._add_to_graph(graph, all_entities, all_relationships[:self.max_relationships])
        
        return graph
    
    def extract_entities(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities with optimized patterns"""
        entities = []
        seen = set()
        
        # Sample first 10 chunks for performance
        chunks_to_process = doc_data['chunks'][:10]
        text = " ".join([c.page_content for c in chunks_to_process])
        
        # Optimized patterns
        patterns = {
            'ORG': [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:LLC|Inc\.|Corp\.|Corporation|Company|Ltd\.|Limited)\b',
                r'\b(Party\s+[A-Z])\b',
            ],
            'DATE': [
                r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
            ],
            'MONEY': [
                r'(\$[\d,]+(?:\.\d{2})?)',
            ],
            'CLAUSE': [
                r'\b((?:Article|Section|Clause)\s+[\dIVX]+(?:\.\d+)*)\b',
            ]
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                for match in matches[:20]:  # Limit per type
                    if isinstance(match, tuple):
                        match = match[0]
                    
                    match = match.strip()
                    if len(match) > 2 and match not in seen:
                        entities.append({
                            'text': match,
                            'type': entity_type,
                            'source': doc_data['name']
                        })
                        seen.add(match)
        
        return entities
    
    def extract_relationships(
        self, 
        doc_data: Dict[str, Any], 
        entities: List[Dict[str, Any]]
    ) -> List[tuple]:
        """Extract simple relationships"""
        relationships = []
        
        # Simple co-occurrence in chunks
        for chunk in doc_data['chunks'][:5]:
            chunk_text = chunk.page_content
            entities_in_chunk = [e['text'] for e in entities if e['text'] in chunk_text]
            
            # Create relationships between co-occurring entities
            for i, ent1 in enumerate(entities_in_chunk[:5]):
                for ent2 in entities_in_chunk[i+1:i+3]:  # Limit pairs
                    relationships.append((ent1, 'MENTIONED_WITH', ent2))
        
        return list(set(relationships))[:50]  # Limit and deduplicate
    
    def _add_to_graph(
        self, 
        graph: nx.DiGraph, 
        entities: List[Dict[str, Any]], 
        relationships: List[tuple]
    ):
        """Add nodes and edges to graph"""
        
        # Add nodes
        for entity in entities:
            node_id = entity['text']
            if node_id not in graph:
                graph.add_node(
                    node_id,
                    type=entity['type'],
                    source=entity['source']
                )
        
        # Add edges
        for ent1, relation, ent2 in relationships:
            if ent1 in graph and ent2 in graph:
                graph.add_edge(ent1, ent2, relation=relation)
    
    def get_graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
        }
        
        # Entity types
        type_counts = defaultdict(int)
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'UNKNOWN')
            type_counts[node_type] += 1
        
        stats['entity_types'] = dict(type_counts)
        
        return stats
