"""
Hybrid Retrieval Strategies
Combines dense (semantic) and sparse (keyword) retrieval methods
"""

from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict


class HybridRetriever:
    """
    Hybrid retriever combining:
    1. Dense retrieval (embeddings/semantic search)
    2. Sparse retrieval (BM25/keyword search)
    3. Hierarchical retrieval (section-aware search)
    """
    
    def __init__(self, vector_store: FAISS, documents: List[Document], top_k: int = 6):
        self.vector_store = vector_store
        self.documents = documents
        self.top_k = top_k
        
        # Build BM25 index
        self.build_bm25_index()
        
        # Build section index
        self.build_section_index()
    
    def build_bm25_index(self):
        """Build BM25 sparse retrieval index"""
        # Tokenize documents
        tokenized_docs = [doc.page_content.lower().split() for doc in self.documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def build_section_index(self):
        """Build index organized by document sections"""
        self.section_index = defaultdict(list)
        
        for doc in self.documents:
            section = doc.metadata.get('section', 'Unknown')
            self.section_index[section].append(doc)
    
    def dense_search(self, query: str, k: int = None) -> List[Document]:
        """
        Dense semantic search using embeddings
        """
        k = k or self.top_k
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def sparse_search(self, query: str, k: int = None) -> List[Document]:
        """
        Sparse keyword search using BM25
        """
        k = k or self.top_k
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Return corresponding documents
        results = [self.documents[i] for i in top_indices if scores[i] > 0]
        
        return results
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Document]:
        """
        Hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query
            k: Number of results to return
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
        """
        k = k or self.top_k
        
        # Get results from both methods
        dense_results = self.dense_search(query, k=k * 2)
        sparse_results = self.sparse_search(query, k=k * 2)
        
        # Score documents
        doc_scores = defaultdict(float)
        
        # Score dense results (inverse rank scoring)
        for rank, doc in enumerate(dense_results):
            doc_id = self._get_doc_id(doc)
            score = (1.0 / (rank + 1)) * dense_weight
            doc_scores[doc_id] += score
        
        # Score sparse results
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        for rank, doc in enumerate(sparse_results):
            doc_id = self._get_doc_id(doc)
            # Normalize BM25 score
            normalized_score = bm25_scores[self.documents.index(doc)] / (max(bm25_scores) + 1e-6)
            score = normalized_score * sparse_weight
            doc_scores[doc_id] += score
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-k documents
        doc_id_to_doc = {self._get_doc_id(doc): doc for doc in self.documents}
        results = [doc_id_to_doc[doc_id] for doc_id, score in sorted_docs[:k]]
        
        return results
    
    def hierarchical_search(self, query: str, k: int = None) -> List[Document]:
        """
        Hierarchical search that considers document structure
        
        Strategy:
        1. Identify relevant sections using semantic search
        2. Search within those sections more deeply
        3. Include parent/sibling sections for context
        """
        k = k or self.top_k
        
        # Step 1: Find relevant sections
        initial_results = self.dense_search(query, k=3)
        
        relevant_sections = set()
        for doc in initial_results:
            relevant_sections.add(doc.metadata.get('section', 'Unknown'))
            # Also add parent section
            parent = self._get_parent_section(doc.metadata.get('section', ''))
            if parent:
                relevant_sections.add(parent)
        
        # Step 2: Search within relevant sections
        section_results = []
        for section in relevant_sections:
            section_docs = self.section_index.get(section, [])
            # Score documents in this section
            for doc in section_docs:
                section_results.append(doc)
        
        # Step 3: Re-rank all section results using hybrid search
        if not section_results:
            return self.hybrid_search(query, k=k)
        
        # Create temporary retriever for section results
        # Re-use the hybrid logic on filtered results
        section_doc_ids = {self._get_doc_id(doc) for doc in section_results}
        
        # Get hybrid results
        all_results = self.hybrid_search(query, k=k * 2)
        
        # Prefer results from relevant sections
        final_results = []
        
        # First add results from relevant sections
        for doc in all_results:
            if self._get_doc_id(doc) in section_doc_ids:
                final_results.append(doc)
                if len(final_results) >= k:
                    break
        
        # Fill remaining with other results
        if len(final_results) < k:
            for doc in all_results:
                if self._get_doc_id(doc) not in section_doc_ids:
                    final_results.append(doc)
                    if len(final_results) >= k:
                        break
        
        return final_results[:k]
    
    def search_by_section(self, section_name: str, query: str = None, k: int = None) -> List[Document]:
        """
        Search within a specific section
        """
        k = k or self.top_k
        
        section_docs = self.section_index.get(section_name, [])
        
        if not section_docs:
            return []
        
        if query:
            # Rank section docs by relevance to query
            # Simple approach: use BM25 scoring
            tokenized_query = query.lower().split()
            
            doc_scores = []
            for doc in section_docs:
                tokens = doc.page_content.lower().split()
                # Simple scoring: count matching terms
                score = sum(1 for token in tokenized_query if token in tokens)
                doc_scores.append((doc, score))
            
            # Sort by score
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in doc_scores[:k]]
        else:
            # Return all docs in section
            return section_docs[:k]
    
    def multi_query_search(self, queries: List[str], k: int = None) -> List[Document]:
        """
        Search using multiple query variations
        Useful for query expansion
        """
        k = k or self.top_k
        
        # Collect results from all queries
        all_results = []
        for query in queries:
            results = self.hybrid_search(query, k=k)
            all_results.extend(results)
        
        # Deduplicate while preserving order of first occurrence
        seen = set()
        unique_results = []
        
        for doc in all_results:
            doc_id = self._get_doc_id(doc)
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
        
        return unique_results[:k]
    
    def _get_doc_id(self, doc: Document) -> str:
        """Generate unique ID for a document"""
        return f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_index', 0)}"
    
    def _get_parent_section(self, section: str) -> str:
        """
        Get parent section from section string
        Example: "1.2.3" -> "1.2"
        """
        if not section or '.' not in section:
            return ""
        
        parts = section.split('.')
        if len(parts) <= 1:
            return ""
        
        return '.'.join(parts[:-1])


class QueryExpander:
    """Expand queries to improve retrieval"""
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into multiple variations
        """
        variations = [query]
        
        # Add variations
        # 1. Question form
        if not query.endswith('?'):
            variations.append(query + '?')
        
        # 2. Statement form
        if query.endswith('?'):
            variations.append(query.rstrip('?'))
        
        # 3. Add common legal phrases
        legal_templates = [
            f"What are the {query}",
            f"Define {query}",
            f"Explain {query}",
            f"{query} provisions",
            f"{query} clause",
        ]
        
        # Only add templates that make sense
        if len(query.split()) <= 3:
            variations.extend(legal_templates[:2])
        
        return list(set(variations))


class ReRanker:
    """Re-rank retrieved documents using cross-encoder or LLM"""
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Re-rank documents based on relevance to query
        
        This is a simple implementation. For production, use:
        - Cross-encoder models
        - LLM-based reranking
        - Learning-to-rank models
        """
        
        # Simple scoring based on term overlap and position
        scored_docs = []
        
        query_terms = set(query.lower().split())
        
        for doc in documents:
            content = doc.page_content.lower()
            content_terms = set(content.split())
            
            # Calculate scores
            term_overlap = len(query_terms & content_terms) / len(query_terms) if query_terms else 0
            
            # Bonus for query terms appearing early in document
            position_score = 0
            for term in query_terms:
                pos = content.find(term)
                if pos >= 0:
                    position_score += 1.0 / (1 + pos / 100)  # Earlier = higher score
            
            # Metadata bonus
            metadata_score = 0
            section = doc.metadata.get('section', '').lower()
            if any(term in section for term in query_terms):
                metadata_score = 0.2
            
            # Combined score
            total_score = term_overlap + (position_score / 10) + metadata_score
            
            scored_docs.append((doc, total_score))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]


# Example usage
def demo_hybrid_retrieval():
    """Demonstrate hybrid retrieval"""
    from langchain.schema import Document
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    # Sample documents
    docs = [
        Document(
            page_content="This contract is effective from January 1, 2024 between Party A and Party B.",
            metadata={'source': 'contract.pdf', 'section': '1.1 Effective Date', 'page': 1}
        ),
        Document(
            page_content="Payment terms: Party A shall pay $50,000 within 30 days of invoice.",
            metadata={'source': 'contract.pdf', 'section': '2.1 Payment Schedule', 'page': 2}
        ),
        Document(
            page_content="Either party may terminate this agreement with 30 days written notice.",
            metadata={'source': 'contract.pdf', 'section': '3.1 Termination', 'page': 3}
        ),
    ]
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Create hybrid retriever
    retriever = HybridRetriever(vector_store, docs, top_k=2)
    
    # Test different retrieval methods
    query = "What are the payment terms?"
    
    print("Query:", query)
    print("\n1. Dense Search:")
    for doc in retriever.dense_search(query):
        print(f"  - {doc.metadata['section']}: {doc.page_content[:50]}...")
    
    print("\n2. Sparse Search (BM25):")
    for doc in retriever.sparse_search(query):
        print(f"  - {doc.metadata['section']}: {doc.page_content[:50]}...")
    
    print("\n3. Hybrid Search:")
    for doc in retriever.hybrid_search(query):
        print(f"  - {doc.metadata['section']}: {doc.page_content[:50]}...")


if __name__ == "__main__":
    # Install required package
    print("Note: Install rank-bm25 with: pip install rank-bm25")
    # demo_hybrid_retrieval()
