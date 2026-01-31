"""
Optimized Hybrid Retrieval Strategies
Faster retrieval with better relevance
"""

from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict


class HybridRetriever:
    """Optimized hybrid retriever"""
    
    def __init__(self, vector_store: FAISS, documents: List[Document], top_k: int = 4):
        self.vector_store = vector_store
        self.documents = documents
        self.top_k = top_k
        
        # Build optimized BM25 index
        self.build_bm25_index()
        
        # Build section index
        self.build_section_index()
    
    def build_bm25_index(self):
        """Build BM25 index with preprocessing"""
        # Preprocess and tokenize
        tokenized_docs = []
        for doc in self.documents:
            # Simple tokenization
            tokens = doc.page_content.lower().split()
            # Remove very short tokens
            tokens = [t for t in tokens if len(t) > 2]
            tokenized_docs.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def build_section_index(self):
        """Build section-based index"""
        self.section_index = defaultdict(list)
        
        for doc in self.documents:
            section = doc.metadata.get('section', 'Unknown')
            self.section_index[section].append(doc)
    
    def dense_search(self, query: str, k: int = None) -> List[Document]:
        """Semantic search using embeddings"""
        k = k or self.top_k
        return self.vector_store.similarity_search(query, k=k)
    
    def sparse_search(self, query: str, k: int = None) -> List[Document]:
        """Keyword search using BM25"""
        k = k or self.top_k
        
        # Tokenize query
        query_tokens = [t for t in query.lower().split() if len(t) > 2]
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Filter by minimum score
        results = [self.documents[i] for i in top_indices if scores[i] > 0]
        
        return results[:k]
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = None,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35
    ) -> List[Document]:
        """
        Optimized hybrid search
        
        Uses reciprocal rank fusion for combining results
        """
        k = k or self.top_k
        
        # Get results from both methods
        dense_results = self.dense_search(query, k=min(k*2, 10))
        sparse_results = self.sparse_search(query, k=min(k*2, 10))
        
        # Reciprocal Rank Fusion
        doc_scores = defaultdict(float)
        
        # Score dense results
        for rank, doc in enumerate(dense_results):
            doc_id = self._get_doc_id(doc)
            # RRF formula: 1 / (k + rank)
            doc_scores[doc_id] += dense_weight * (1.0 / (60 + rank))
        
        # Score sparse results  
        for rank, doc in enumerate(sparse_results):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] += sparse_weight * (1.0 / (60 + rank))
        
        # Sort and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        doc_id_to_doc = {self._get_doc_id(doc): doc for doc in self.documents}
        results = [doc_id_to_doc[doc_id] for doc_id, _ in sorted_docs[:k]]
        
        return results
    
    def hierarchical_search(self, query: str, k: int = None) -> List[Document]:
        """Section-aware hierarchical search"""
        k = k or self.top_k
        
        # Get initial results
        initial_results = self.dense_search(query, k=min(k, 5))
        
        if not initial_results:
            return self.hybrid_search(query, k)
        
        # Find relevant sections
        relevant_sections = set()
        for doc in initial_results:
            section = doc.metadata.get('section', 'Unknown')
            relevant_sections.add(section)
        
        # Get more docs from relevant sections
        section_docs = []
        for section in relevant_sections:
            section_docs.extend(self.section_index.get(section, []))
        
        # Deduplicate and limit
        seen_ids = set()
        unique_docs = []
        
        for doc in initial_results + section_docs:
            doc_id = self._get_doc_id(doc)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        return unique_docs[:k]
    
    def _get_doc_id(self, doc: Document) -> str:
        """Generate document ID"""
        return f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_index', 0)}"


class QueryExpander:
    """Simple query expansion"""
    
    def expand_query(self, query: str) -> List[str]:
        """Generate query variations"""
        variations = [query]
        
        # Add question mark if not present
        if '?' not in query:
            variations.append(query + '?')
        
        # Remove question mark if present
        if query.endswith('?'):
            variations.append(query.rstrip('?'))
        
        return variations[:3]  # Limit variations
