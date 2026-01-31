"""
Knowledge Graph Builder
Extracts entities and relationships from documents
"""

import re
from typing import List, Dict, Any, Tuple
import networkx as nx
from collections import defaultdict
import spacy


class KnowledgeGraphBuilder:
    """
    Build knowledge graph from documents
    Extracts entities and relationships
    """
    
    def __init__(self, use_spacy: bool = False):
        self.use_spacy = use_spacy
        self.nlp = None
        
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Warning: spaCy model not found. Using rule-based extraction.")
                print("Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
    
    def build_from_documents(self, documents: Dict[str, Any]) -> nx.DiGraph:
        """
        Build knowledge graph from processed documents
        
        Args:
            documents: Dictionary of processed documents
            
        Returns:
            NetworkX directed graph
        """
        graph = nx.DiGraph()
        
        for doc_name, doc_data in documents.items():
            # Extract entities and relationships from each document
            entities = self.extract_entities(doc_data)
            relationships = self.extract_relationships(doc_data, entities)
            
            # Add to graph
            self._add_to_graph(graph, entities, relationships, doc_name)
        
        return graph
    
    def extract_entities(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract named entities from document
        
        Entity types:
        - PERSON: People, parties
        - ORG: Organizations, companies
        - DATE: Dates
        - MONEY: Monetary amounts
        - GPE: Countries, cities
        - LAW: Legal references
        - CLAUSE: Contract clauses
        """
        entities = []
        
        if self.use_spacy and self.nlp:
            entities = self._extract_with_spacy(doc_data)
        else:
            entities = self._extract_with_rules(doc_data)
        
        return entities
    
    def _extract_with_spacy(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER"""
        entities = []
        seen_entities = set()
        
        # Process each chunk
        for chunk in doc_data['chunks'][:10]:  # Limit for performance
            doc = self.nlp(chunk.page_content)
            
            for ent in doc.ents:
                entity_key = (ent.text, ent.label_)
                
                if entity_key not in seen_entities:
                    entities.append({
                        'text': ent.text,
                        'type': ent.label_,
                        'source': doc_data['name'],
                        'section': chunk.metadata.get('section', 'Unknown')
                    })
                    seen_entities.add(entity_key)
        
        return entities
    
    def _extract_with_rules(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities using rule-based patterns"""
        entities = []
        seen_entities = set()
        
        # Combine text from chunks
        full_text = " ".join([chunk.page_content for chunk in doc_data['chunks'][:20]])
        
        # Pattern matching for different entity types
        patterns = {
            'PERSON': [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b',
            ],
            'ORG': [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:LLC|Inc\.|Corp\.|Corporation|Company|Ltd\.|Limited|LLP)\b',
                r'\b(Party\s+[A-Z])\b',
            ],
            'DATE': [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
                r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
                r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            ],
            'MONEY': [
                r'(\$[\d,]+(?:\.\d{2})?)',
                r'\b(USD\s+[\d,]+(?:\.\d{2})?)\b',
            ],
            'CLAUSE': [
                r'\b((?:Article|Section|Clause)\s+[\dIVXLCDM]+(?:\.\d+)*)\b',
            ],
            'LAW': [
                r'\b([A-Z][a-z]+\s+Act(?:\s+of\s+\d{4})?)\b',
                r'\b(U\.?S\.?C\.?\s+§?\s*\d+)\b',
            ]
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    
                    entity_key = (match.strip(), entity_type)
                    
                    if entity_key not in seen_entities and len(match.strip()) > 2:
                        entities.append({
                            'text': match.strip(),
                            'type': entity_type,
                            'source': doc_data['name']
                        })
                        seen_entities.add(entity_key)
        
        return entities
    
    def extract_relationships(
        self, 
        doc_data: Dict[str, Any], 
        entities: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between entities
        
        Returns:
            List of (entity1, relation, entity2) tuples
        """
        relationships = []
        
        # Combine text
        full_text = " ".join([chunk.page_content for chunk in doc_data['chunks'][:20]])
        
        # Define relationship patterns
        # Format: (pattern, entity1_type, relation, entity2_type)
        relation_patterns = [
            (r'(\w+(?:\s+\w+)*)\s+(?:shall|will|must)\s+pay\s+(\w+(?:\s+\w+)*)\s+(\$[\d,]+)', 
             'ORG', 'PAYS', 'ORG'),
            
            (r'(?:between|by and between)\s+(.+?)\s+and\s+(.+?)(?:\.|,)',
             'ORG', 'CONTRACT_WITH', 'ORG'),
            
            (r'(\w+(?:\s+\w+)*)\s+agrees?\s+to\s+(.+?)(?:\.|,)',
             'ORG', 'AGREES_TO', 'ACTION'),
            
            (r'effective\s+(?:as of|from|on)\s+(.+?)(?:\.|,)',
             'CONTRACT', 'EFFECTIVE_DATE', 'DATE'),
            
            (r'(?:terminate|termination).*?(\d+)\s+days',
             'CONTRACT', 'TERMINATION_PERIOD', 'DURATION'),
        ]
        
        # Extract relationships using patterns
        for pattern, ent1_type, relation, ent2_type in relation_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    ent1 = match.group(1).strip()
                    ent2 = match.group(2).strip() if len(match.groups()) >= 2 else ""
                    
                    if ent1 and ent2:
                        relationships.append((ent1, relation, ent2))
        
        # Entity co-occurrence relationships
        # If two entities appear in the same chunk, create a MENTIONED_WITH relationship
        entity_texts = {e['text']: e for e in entities}
        
        for chunk in doc_data['chunks'][:10]:
            chunk_text = chunk.page_content
            entities_in_chunk = [
                e['text'] for e in entities 
                if e['text'] in chunk_text
            ]
            
            # Create relationships between entities in the same chunk
            for i, ent1 in enumerate(entities_in_chunk):
                for ent2 in entities_in_chunk[i+1:]:
                    relationships.append((ent1, 'MENTIONED_WITH', ent2))
        
        return list(set(relationships))  # Deduplicate
    
    def _add_to_graph(
        self, 
        graph: nx.DiGraph, 
        entities: List[Dict[str, Any]], 
        relationships: List[Tuple[str, str, str]],
        doc_name: str
    ):
        """Add entities and relationships to the graph"""
        
        # Add entity nodes
        for entity in entities:
            node_id = entity['text']
            
            if node_id not in graph:
                graph.add_node(
                    node_id,
                    type=entity['type'],
                    sources=[doc_name]
                )
            else:
                # Add document to sources if not already there
                if doc_name not in graph.nodes[node_id].get('sources', []):
                    graph.nodes[node_id]['sources'].append(doc_name)
        
        # Add relationship edges
        for ent1, relation, ent2 in relationships:
            # Check if both entities exist in graph
            if ent1 in graph and ent2 in graph:
                graph.add_edge(
                    ent1,
                    ent2,
                    relation=relation,
                    source=doc_name
                )
    
    def query_graph(self, graph: nx.DiGraph, query_type: str, **kwargs) -> Any:
        """
        Query the knowledge graph
        
        Query types:
        - 'find_entity': Find entity by name or type
        - 'find_relationships': Find all relationships for an entity
        - 'find_path': Find path between two entities
        - 'find_neighbors': Find neighboring entities
        """
        
        if query_type == 'find_entity':
            entity_name = kwargs.get('name')
            entity_type = kwargs.get('type')
            
            results = []
            for node in graph.nodes():
                node_data = graph.nodes[node]
                
                if entity_name and entity_name.lower() in node.lower():
                    results.append({'name': node, 'data': node_data})
                elif entity_type and node_data.get('type') == entity_type:
                    results.append({'name': node, 'data': node_data})
            
            return results
        
        elif query_type == 'find_relationships':
            entity_name = kwargs.get('name')
            
            if entity_name not in graph:
                return []
            
            relationships = []
            
            # Outgoing edges
            for target in graph.successors(entity_name):
                edge_data = graph.edges[entity_name, target]
                relationships.append({
                    'from': entity_name,
                    'to': target,
                    'relation': edge_data.get('relation', 'UNKNOWN'),
                    'type': 'outgoing'
                })
            
            # Incoming edges
            for source in graph.predecessors(entity_name):
                edge_data = graph.edges[source, entity_name]
                relationships.append({
                    'from': source,
                    'to': entity_name,
                    'relation': edge_data.get('relation', 'UNKNOWN'),
                    'type': 'incoming'
                })
            
            return relationships
        
        elif query_type == 'find_path':
            start = kwargs.get('start')
            end = kwargs.get('end')
            
            if start not in graph or end not in graph:
                return None
            
            try:
                path = nx.shortest_path(graph, start, end)
                
                # Get relationships along path
                path_with_relations = []
                for i in range(len(path) - 1):
                    edge_data = graph.edges[path[i], path[i+1]]
                    path_with_relations.append({
                        'from': path[i],
                        'to': path[i+1],
                        'relation': edge_data.get('relation', 'UNKNOWN')
                    })
                
                return path_with_relations
            except nx.NetworkXNoPath:
                return None
        
        elif query_type == 'find_neighbors':
            entity_name = kwargs.get('name')
            max_distance = kwargs.get('max_distance', 1)
            
            if entity_name not in graph:
                return []
            
            # BFS to find neighbors within max_distance
            neighbors = set()
            queue = [(entity_name, 0)]
            visited = {entity_name}
            
            while queue:
                current, distance = queue.pop(0)
                
                if distance < max_distance:
                    for neighbor in graph.neighbors(current):
                        if neighbor not in visited:
                            neighbors.add(neighbor)
                            visited.add(neighbor)
                            queue.append((neighbor, distance + 1))
            
            return list(neighbors)
        
        return None
    
    def get_graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'num_connected_components': nx.number_weakly_connected_components(graph),
        }
        
        # Entity type distribution
        type_counts = defaultdict(int)
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'UNKNOWN')
            type_counts[node_type] += 1
        
        stats['entity_types'] = dict(type_counts)
        
        # Relation type distribution
        relation_counts = defaultdict(int)
        for edge in graph.edges():
            relation = graph.edges[edge].get('relation', 'UNKNOWN')
            relation_counts[relation] += 1
        
        stats['relation_types'] = dict(relation_counts)
        
        # Most connected entities (top 5)
        degree_centrality = nx.degree_centrality(graph)
        top_entities = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        stats['most_connected_entities'] = [
            {'entity': entity, 'centrality': score} 
            for entity, score in top_entities
        ]
        
        return stats


# Example usage
def demo_knowledge_graph():
    """Demonstrate knowledge graph construction"""
    from langchain.schema import Document
    
    # Sample document data
    doc_data = {
        'name': 'sample_contract.pdf',
        'chunks': [
            Document(
                page_content="""
                This Agreement is made on January 15, 2024, between ABC Corporation 
                and XYZ Limited. ABC Corporation agrees to pay XYZ Limited the sum 
                of $100,000 within 30 days.
                """,
                metadata={'section': '1. Parties', 'page': 1}
            ),
            Document(
                page_content="""
                Either party may terminate this agreement with 30 days written notice.
                The termination clause is outlined in Section 5.
                """,
                metadata={'section': '5. Termination', 'page': 3}
            ),
        ],
        'structure': {},
        'metadata': {}
    }
    
    # Build knowledge graph
    kg_builder = KnowledgeGraphBuilder(use_spacy=False)
    graph = kg_builder.build_from_documents({'sample_contract.pdf': doc_data})
    
    # Print statistics
    stats = kg_builder.get_graph_statistics(graph)
    print("Knowledge Graph Statistics:")
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Entity Types: {stats['entity_types']}")
    print(f"Relation Types: {stats['relation_types']}")
    
    # Query examples
    print("\nQuerying for ORG entities:")
    orgs = kg_builder.query_graph(graph, 'find_entity', type='ORG')
    for org in orgs:
        print(f"  - {org['name']}")
    
    if orgs:
        print(f"\nRelationships for {orgs[0]['name']}:")
        rels = kg_builder.query_graph(graph, 'find_relationships', name=orgs[0]['name'])
        for rel in rels:
            print(f"  {rel['from']} --[{rel['relation']}]--> {rel['to']}")


if __name__ == "__main__":
    demo_knowledge_graph()
