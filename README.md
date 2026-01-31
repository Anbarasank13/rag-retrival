# Hierarchical RAG for Legal Document Analysis

A sophisticated Retrieval-Augmented Generation (RAG) system designed for multi-document intelligence with hierarchical structure extraction, knowledge graph construction, and advanced reasoning capabilities.

## 🎯 Project Overview

This capstone project implements a **Hierarchical RAG system** specifically tailored for legal document analysis. It goes beyond basic RAG by incorporating:

- **Hierarchical Document Processing**: Automatically extracts document structure (sections, subsections, clauses)
- **Multi-Document Intelligence**: Compare and analyze multiple documents simultaneously
- **Knowledge Graph Construction**: Build entity-relationship graphs from documents
- **Hybrid Retrieval**: Combines dense (semantic) and sparse (keyword-based) search
- **Clause Extraction**: Automatically identifies and categorizes legal clauses
- **Structured Reasoning**: Multi-hop reasoning with citation tracking

## 🚀 Features

### 1. **Document Processing**
- Automatic section hierarchy detection
- Support for PDF and TXT files
- Metadata extraction (parties, dates, amounts)
- Hierarchical chunk creation with context preservation

### 2. **Advanced Retrieval**
- **Dense Retrieval**: Semantic search using embeddings
- **Sparse Retrieval**: BM25-based keyword matching
- **Hybrid Search**: Weighted combination of dense + sparse
- **Hierarchical Search**: Section-aware retrieval
- **Query Expansion**: Multiple query variations for better recall
- **Re-ranking**: Relevance-based result refinement

### 3. **Knowledge Graph**
- Entity extraction (persons, organizations, dates, amounts)
- Relationship detection between entities
- Interactive graph visualization
- Entity-based queries and navigation

### 4. **Legal Clause Analysis**
Automatically detects and extracts:
- Termination clauses
- Payment terms
- Confidentiality agreements
- Liability limitations
- Intellectual property rights
- Governing law provisions
- Dispute resolution mechanisms
- Force majeure clauses
- Warranties
- Term duration

### 5. **Multi-Document Comparison**
- Structure comparison
- Clause-by-clause analysis
- Entity comparison
- Content similarity detection
- Side-by-side visualization

### 6. **Intelligent Reasoning**
- Multi-step reasoning with citations
- Query decomposition
- Source attribution with section/page numbers
- Confidence scoring

## 📁 Project Structure

```
hierarchical-rag/
├── app_hierarchical.py          # Main Streamlit application
├── document_processor.py        # Hierarchical document processing
├── retrieval_strategies.py      # Hybrid retrieval implementations
├── knowledge_graph.py           # Knowledge graph construction
├── clause_extractor.py          # Legal clause extraction
├── comparison_engine.py         # Document comparison tools
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Setup Steps

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd hierarchical-rag
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model (optional, for better entity extraction)**
```bash
python -m spacy download en_core_web_sm
```

5. **Create .env file (optional)**
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

## 🎮 Usage

### Running the Application

```bash
streamlit run app_hierarchical.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Enter API Key**: Add your Google Gemini API key in the sidebar
2. **Upload Documents**: Upload one or more PDF/TXT files
3. **Process Documents**: Click "Process Documents" to analyze
4. **Ask Questions**: Use the chat interface or quick actions
5. **Explore Features**: Navigate through tabs for different analyses

### Example Queries

**Single Document Analysis:**
- "What are the termination conditions?"
- "List all payment terms with amounts"
- "Who are the parties involved?"
- "When does this agreement become effective?"

**Multi-Document Comparison:**
- "Compare payment terms across all documents"
- "Which contract has the longest termination period?"
- "Find inconsistencies in confidentiality clauses"
- "What are the common sections across documents?"

**Advanced Reasoning:**
- "If Party A terminates early, what are the total costs?"
- "Trace all obligations of Party B across documents"
- "What happens if payment is 45 days late?"

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│                   (Streamlit App)                        │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              Document Processing Layer                   │
│  • Hierarchical Structure Extraction                    │
│  • Metadata Extraction                                  │
│  • Section Navigation                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│               Retrieval Layer                           │
│  • Dense Search (Embeddings)                            │
│  • Sparse Search (BM25)                                 │
│  • Hybrid Search                                        │
│  • Hierarchical Search                                  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              Analysis Layer                             │
│  • Knowledge Graph                                      │
│  • Clause Extraction                                    │
│  • Document Comparison                                  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│             Generation Layer                            │
│  • Multi-hop Reasoning                                  │
│  • Citation Management                                  │
│  • Response Synthesis                                   │
└─────────────────────────────────────────────────────────┘
```

### Key Algorithms

**1. Hierarchical Chunking**
- Preserves document structure in chunks
- Adds parent-child relationships
- Maintains section context

**2. Hybrid Retrieval**
```
Score(doc) = α × DenseScore(doc) + β × SparseScore(doc)
where α + β = 1
```

**3. Knowledge Graph Construction**
- Rule-based entity extraction
- Pattern-based relationship detection
- Co-occurrence analysis

## 📊 Evaluation Metrics

The system can be evaluated on:

1. **Retrieval Quality**
   - Precision@K
   - Recall@K
   - MRR (Mean Reciprocal Rank)

2. **Answer Quality**
   - Accuracy (human evaluation)
   - Citation accuracy
   - Completeness

3. **System Performance**
   - Response time
   - Throughput
   - Resource usage

## 🔧 Configuration Options

### Sidebar Settings

- **Retrieval Strategy**: Choose between Hybrid, Dense, or Hierarchical
- **Top K Results**: Number of chunks to retrieve (3-10)
- **Enable Reranking**: Refine results using cross-encoder
- **Show Reasoning**: Display step-by-step reasoning
- **Build Knowledge Graph**: Enable entity extraction
- **Extract Entities**: Enable named entity recognition

### Advanced Configuration

Edit constants in `app_hierarchical.py`:
```python
CHUNK_SIZE = 1000          # Size of text chunks
CHUNK_OVERLAP = 200        # Overlap between chunks
EMBEDDING_MODEL = "..."    # Embedding model
RETRIEVER_K = 6           # Default retrieval count
```

## 🧪 Testing

### Running Individual Modules

Each module has a demo function:

```bash
# Test document processor
python document_processor.py

# Test retrieval strategies
python retrieval_strategies.py

# Test knowledge graph
python knowledge_graph.py

# Test clause extractor
python clause_extractor.py

# Test comparison engine
python comparison_engine.py
```

### Sample Test Documents

Create sample legal documents to test:
1. Service Agreement
2. Employment Contract
3. NDA (Non-Disclosure Agreement)
4. License Agreement

## 📈 Future Enhancements

Potential improvements for extended work:

1. **Fine-tuned Models**
   - Custom embedding models for legal text
   - Domain-specific LLMs

2. **Advanced Features**
   - Contract risk analysis
   - Automated clause generation
   - Version control for documents
   - Workflow automation

3. **Integration**
   - Database backends
   - Document management systems
   - Legal research platforms

4. **Evaluation**
   - Automated benchmark suite
   - A/B testing framework
   - User study platform

## 🎓 Academic Context

This project demonstrates:

- **Information Retrieval**: Hybrid search, re-ranking
- **Natural Language Processing**: NER, relationship extraction
- **Knowledge Representation**: Graph structures
- **Software Engineering**: Modular architecture, clean code
- **Domain Expertise**: Legal document understanding

Suitable for:
- Computer Science capstone
- Information Systems project
- AI/ML thesis work
- Legal Tech demonstration

## 📝 Documentation

### API Documentation

Each module is well-documented with:
- Function docstrings
- Type hints
- Usage examples
- Edge case handling

### Code Quality

- PEP 8 compliant
- Modular design
- Error handling
- Performance optimizations

## 🤝 Contributing

To extend this project:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit pull request

## 📄 License

This project is created for educational purposes.

## 🙏 Acknowledgments

- LangChain for RAG framework
- Google for Gemini API
- Streamlit for UI framework
- spaCy for NLP capabilities

## 📧 Contact

For questions or collaboration:
- Create an issue on GitHub
- Email: [your-email]

## 🎯 Capstone Presentation Tips

### What to Highlight

1. **Problem Statement**
   - Why hierarchical RAG for legal docs?
   - Limitations of basic RAG

2. **Technical Innovation**
   - Hierarchical structure extraction
   - Hybrid retrieval approach
   - Knowledge graph construction

3. **Implementation**
   - System architecture
   - Key algorithms
   - Code quality

4. **Results**
   - Demo with real documents
   - Performance metrics
   - User feedback

5. **Future Work**
   - Scalability improvements
   - Additional features
   - Commercial applications

### Demo Script

1. Upload multiple contracts
2. Show hierarchical structure extraction
3. Compare clauses across documents
4. Visualize knowledge graph
5. Ask complex multi-hop question
6. Show reasoning steps with citations

---

**Built with ❤️ for advancing legal document intelligence**
#   r a g - r e t r i v a l  
 