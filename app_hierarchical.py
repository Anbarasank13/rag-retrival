import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import asyncio
import tempfile
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import time
from typing import List, Dict, Any
import json
import re
from collections import defaultdict
import networkx as nx
from pyvis.network import Network
import pandas as pd

# Import custom modules
from document_processor import HierarchicalDocumentProcessor
from knowledge_graph import KnowledgeGraphBuilder
from retrieval_strategies import HybridRetriever
from clause_extractor import ClauseExtractor
from comparison_engine import DocumentComparator

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/text-embedding-004"
RETRIEVER_K = 6
DEFAULT_SYSTEM_MESSAGE = """
You are an Advanced Hierarchical RAG Assistant for Legal Document Analysis 📄⚖️.

Your capabilities:
1. Analyze document structure and hierarchy
2. Extract and compare clauses across multiple documents
3. Perform multi-hop reasoning with citation
4. Build knowledge graphs of entities and relationships
5. Provide structured, well-reasoned answers with sources

Follow these rules:
- Always cite sources with section numbers and page references
- For complex questions, break down your reasoning into steps
- When comparing documents, provide side-by-side analysis
- Highlight key entities, dates, and obligations
- If uncertain, say so and explain why
"""

# Load environment variables
load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents" not in st.session_state:
        st.session_state.documents = {}
    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = None
    if "document_structures" not in st.session_state:
        st.session_state.document_structures = {}
    if "hybrid_retriever" not in st.session_state:
        st.session_state.hybrid_retriever = None
    if "clause_extractor" not in st.session_state:
        st.session_state.clause_extractor = None


def configure_page():
    st.set_page_config(
        page_title="Hierarchical RAG - Legal Assistant",
        page_icon="⚖️",
        layout="wide",
    )

    st.title("⚖️ Hierarchical RAG Legal Document Assistant")
    st.markdown("### Advanced Multi-Document Intelligence & Structured Reasoning")


def apply_custom_css():
    st.markdown(
        """
        <style>
        /* Main container styling */
        .block-container {
            padding-top: 2rem;
            max-width: 1400px;
        }
        
        /* Document card styling */
        .doc-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Section header styling */
        .section-header {
            background: #f0f2f6;
            padding: 0.8rem;
            border-left: 4px solid #667eea;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        /* Citation styling */
        .citation {
            background: #e3f2fd;
            padding: 0.5rem;
            border-left: 3px solid #2196f3;
            margin: 0.5rem 0;
            border-radius: 3px;
            font-size: 0.9em;
        }
        
        /* Comparison table styling */
        .comparison-table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }
        
        .comparison-table th {
            background: #667eea;
            color: white;
            padding: 0.8rem;
        }
        
        .comparison-table td {
            padding: 0.8rem;
            border: 1px solid #ddd;
        }
        
        /* Entity badge */
        .entity-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            margin: 0.2rem;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .entity-person {
            background: #e1f5fe;
            color: #01579b;
        }
        
        .entity-org {
            background: #f3e5f5;
            color: #4a148c;
        }
        
        .entity-date {
            background: #fff3e0;
            color: #e65100;
        }
        
        .entity-money {
            background: #e8f5e9;
            color: #1b5e20;
        }
        
        /* Progress indicator */
        .reasoning-step {
            background: #fff9c4;
            padding: 0.8rem;
            margin: 0.5rem 0;
            border-left: 4px solid #fbc02d;
            border-radius: 5px;
        }
        
        /* Metrics cards */
        .metric-card {
            background: white;
            padding: 1.2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def handle_sidebar():
    """Enhanced sidebar with more controls"""
    st.sidebar.header("🔑 Configuration")

    # API Key
    api_key = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Enter your API key...",
        value=st.session_state.get("api_key", ""),
    )
    
    if api_key:
        st.session_state.api_key = api_key
        if len(api_key) < 20:
            st.sidebar.error("⚠️ API key looks too short")
        elif not api_key.startswith("AIza"):
            st.sidebar.warning("⚠️ Doesn't look like a Google API key")
        else:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.sidebar.success("✅ API key configured")
    else:
        st.sidebar.info("💡 Enter API key to start")

    st.sidebar.divider()

    # Model Selection
    selected_model = st.sidebar.selectbox(
        "Generation Model",
        [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-image-preview",
            "gemini-live-2.5-flash-preview",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            "gemini-2.0-flash-live-001",
            "gemini-2.0-flash-live-preview-04-09",
            "gemini-2.0-flash-preview-image-generation",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        index=0,
    )
    st.session_state.model = selected_model

    st.sidebar.divider()

    # Retrieval Settings
    st.sidebar.subheader("⚙️ Retrieval Settings")
    
    retrieval_mode = st.sidebar.radio(
        "Retrieval Strategy",
        ["Hybrid (Dense + Sparse)", "Dense Only", "Hierarchical"],
        help="Choose how to retrieve relevant content"
    )
    st.session_state.retrieval_mode = retrieval_mode
    
    top_k = st.sidebar.slider("Top K Results", 3, 10, 6)
    st.session_state.top_k = top_k
    
    enable_reranking = st.sidebar.checkbox("Enable Reranking", value=True)
    st.session_state.enable_reranking = enable_reranking

    st.sidebar.divider()

    # Feature Toggles
    st.sidebar.subheader("✨ Features")
    
    show_reasoning = st.sidebar.checkbox("Show Reasoning Steps", value=True)
    st.session_state.show_reasoning = show_reasoning
    
    enable_kg = st.sidebar.checkbox("Build Knowledge Graph", value=True)
    st.session_state.enable_kg = enable_kg
    
    extract_entities = st.sidebar.checkbox("Extract Entities", value=True)
    st.session_state.extract_entities = extract_entities

    st.sidebar.divider()

    # Document Management
    st.sidebar.subheader("📚 Document Management")
    
    num_docs = len(st.session_state.documents)
    st.sidebar.metric("Documents Loaded", num_docs)
    
    if num_docs > 0:
        st.sidebar.success(f"✅ {num_docs} document(s) ready")
        
        if st.sidebar.button("🗑️ Clear All Documents", use_container_width=True):
            st.session_state.documents = {}
            st.session_state.document_structures = {}
            st.session_state.knowledge_graph = None
            st.session_state.hybrid_retriever = None
            st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
            st.rerun()

    st.sidebar.divider()

    # Chat Controls
    st.sidebar.subheader("💬 Chat Controls")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
            st.rerun()
    
    with col2:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")

    # Export Chat
    message_count = len(st.session_state.messages) - 1
    if message_count > 0:
        st.sidebar.divider()
        chat_text = ""
        for msg in st.session_state.messages[1:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            chat_text += f"{role}: {msg.content}\n\n"
        
        st.sidebar.download_button(
            "📥 Download Chat",
            chat_text,
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            use_container_width=True,
        )

    return selected_model, st.session_state.get("api_key")


def render_document_upload_section():
    """Multi-document upload interface"""
    st.subheader("📁 Document Upload & Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload Legal Documents (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload one or more documents for analysis"
        )
    
    with col2:
        if uploaded_files:
            st.info(f"📊 {len(uploaded_files)} file(s) selected")
    
    return uploaded_files


def process_documents(uploaded_files):
    """Process uploaded documents with hierarchical structure extraction"""
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one document")
        return
    
    user_api_key = st.session_state.get("api_key", "")
    if not user_api_key:
        st.error("❌ Please enter your API key in the sidebar")
        return
    
    with st.spinner("🔄 Processing documents..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            processor = HierarchicalDocumentProcessor(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
            total_files = len(uploaded_files)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                progress = (idx + 1) / total_files
                status_text.text(f"📄 Processing {uploaded_file.name} ({idx + 1}/{total_files})...")
                progress_bar.progress(progress * 0.7)
                
                # Save temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=f".{uploaded_file.name.split('.')[-1]}"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load document
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = TextLoader(tmp_file_path)
                
                documents = loader.load()
                
                # Process with hierarchy extraction
                doc_data = processor.process_document(
                    documents, 
                    uploaded_file.name
                )
                
                st.session_state.documents[uploaded_file.name] = doc_data
                
                os.unlink(tmp_file_path)
            
            # Build hybrid retriever
            status_text.text("🧠 Building retrieval system...")
            progress_bar.progress(0.8)
            
            all_chunks = []
            for doc_data in st.session_state.documents.values():
                all_chunks.extend(doc_data['chunks'])
            
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
            vector_store = FAISS.from_documents(all_chunks, embeddings)
            
            hybrid_retriever = HybridRetriever(
                vector_store=vector_store,
                documents=all_chunks,
                top_k=st.session_state.get('top_k', 6)
            )
            
            st.session_state.hybrid_retriever = hybrid_retriever
            
            # Build knowledge graph if enabled
            if st.session_state.get('enable_kg', True):
                status_text.text("🕸️ Building knowledge graph...")
                progress_bar.progress(0.9)
                
                kg_builder = KnowledgeGraphBuilder()
                knowledge_graph = kg_builder.build_from_documents(
                    st.session_state.documents
                )
                st.session_state.knowledge_graph = knowledge_graph
            
            # Initialize clause extractor
            st.session_state.clause_extractor = ClauseExtractor()
            
            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ Successfully processed {total_files} document(s)!")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def render_document_overview():
    """Display overview of processed documents"""
    if not st.session_state.documents:
        return
    
    st.subheader("📊 Document Overview")
    
    cols = st.columns(min(len(st.session_state.documents), 3))
    
    for idx, (doc_name, doc_data) in enumerate(st.session_state.documents.items()):
        with cols[idx % 3]:
            with st.container():
                st.markdown(
                    f"""
                    <div class="doc-card">
                        <h4>📄 {doc_name}</h4>
                        <p><strong>Sections:</strong> {len(doc_data.get('structure', {}).get('sections', []))}</p>
                        <p><strong>Chunks:</strong> {len(doc_data['chunks'])}</p>
                        <p><strong>Pages:</strong> {doc_data.get('metadata', {}).get('total_pages', 'N/A')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def render_quick_actions():
    """Render quick action buttons for common queries"""
    if not st.session_state.documents:
        return
    
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Find Termination Clauses", use_container_width=True):
            st.session_state.quick_query = "Find and summarize all termination clauses in the documents"
    
    with col2:
        if st.button("💰 Extract Payment Terms", use_container_width=True):
            st.session_state.quick_query = "Extract and compare payment terms across all documents"
    
    with col3:
        if st.button("📅 List Key Dates", use_container_width=True):
            st.session_state.quick_query = "List all important dates mentioned in the documents"
    
    with col4:
        if st.button("⚠️ Identify Risks", use_container_width=True):
            st.session_state.quick_query = "Identify potential legal risks and liabilities mentioned"
    
    # Handle quick query
    if 'quick_query' in st.session_state and st.session_state.quick_query:
        query = st.session_state.quick_query
        del st.session_state.quick_query
        return query
    
    return None


def render_knowledge_graph():
    """Render knowledge graph visualization"""
    if not st.session_state.get('knowledge_graph'):
        return
    
    with st.expander("🕸️ Knowledge Graph Visualization", expanded=False):
        kg = st.session_state.knowledge_graph
        
        if kg and len(kg.nodes()) > 0:
            # Create pyvis network
            net = Network(
                height="500px",
                width="100%",
                bgcolor="#ffffff",
                font_color="#000000"
            )
            
            # Add nodes and edges
            for node in kg.nodes():
                node_data = kg.nodes[node]
                net.add_node(
                    node,
                    label=node,
                    title=f"Type: {node_data.get('type', 'Unknown')}",
                    color=get_node_color(node_data.get('type', 'Unknown'))
                )
            
            for edge in kg.edges():
                edge_data = kg.edges[edge]
                net.add_edge(
                    edge[0],
                    edge[1],
                    title=edge_data.get('relation', ''),
                    label=edge_data.get('relation', '')
                )
            
            # Save and display
            net.save_graph("kg.html")
            with open("kg.html", "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=500)
        else:
            st.info("No entities extracted yet. Process documents to build knowledge graph.")


def get_node_color(node_type):
    """Get color for knowledge graph nodes based on type"""
    colors = {
        'PERSON': '#90caf9',
        'ORG': '#ce93d8',
        'DATE': '#ffcc80',
        'MONEY': '#a5d6a7',
        'GPE': '#ef9a9a',
        'LAW': '#fff59d',
        'CLAUSE': '#80deea',
        'DEFAULT': '#b0bec5'
    }
    return colors.get(node_type, colors['DEFAULT'])


def display_chat_interface():
    """Enhanced chat interface with reasoning display"""
    for message in st.session_state.messages[1:]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # Parse and display structured response
                content = message.content
                
                # Check for reasoning steps
                if "**Reasoning Steps:**" in content:
                    parts = content.split("**Reasoning Steps:**")
                    st.write(parts[0])
                    
                    st.markdown("**Reasoning Steps:**")
                    reasoning_part = parts[1].split("**Answer:**")[0] if "**Answer:**" in parts[1] else parts[1]
                    st.markdown(reasoning_part)
                    
                    if "**Answer:**" in content:
                        answer_part = content.split("**Answer:**")[1]
                        st.markdown("**Answer:**")
                        st.markdown(answer_part)
                else:
                    st.write(content)


@st.cache_resource()
def get_chat_model(model_name: str, api_key_keyed: str):
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.2)


def handle_user_query(chat_model, query=None):
    """Enhanced query handling with multi-hop reasoning"""
    
    # Get query from quick actions or chat input
    if query is None:
        query = st.chat_input(
            "Ask a question about your documents...",
            disabled=(chat_model is None)
        )
    
    if not query or not query.strip():
        return
    
    st.session_state.messages.append(HumanMessage(content=query))
    
    with st.chat_message("user"):
        st.write(query)
    
    # Check if retriever exists
    if not st.session_state.get('hybrid_retriever'):
        with st.chat_message("assistant"):
            error_msg = "❌ Please process documents first"
            st.error(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))
        return
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing documents..."):
            try:
                # Retrieve relevant chunks
                retriever = st.session_state.hybrid_retriever
                
                if st.session_state.get('retrieval_mode') == "Hybrid (Dense + Sparse)":
                    retrieved_docs = retriever.hybrid_search(query)
                elif st.session_state.get('retrieval_mode') == "Hierarchical":
                    retrieved_docs = retriever.hierarchical_search(query)
                else:
                    retrieved_docs = retriever.dense_search(query)
                
                if not retrieved_docs:
                    no_context_msg = "🤷‍♂️ No relevant information found"
                    st.warning(no_context_msg)
                    st.session_state.messages.append(AIMessage(content=no_context_msg))
                    return
                
                # Build context with metadata
                context = build_rich_context(retrieved_docs)
                
                # Create enhanced prompt
                prompt_template = create_reasoning_prompt(
                    show_reasoning=st.session_state.get('show_reasoning', True)
                )
                
                # Generate response
                chain = (
                    RunnableParallel({
                        "context": RunnableLambda(lambda x: context),
                        "question": RunnablePassthrough()
                    })
                    | prompt_template
                    | chat_model
                    | StrOutputParser()
                )
                
                message_placeholder = st.empty()
                full_response = ""
                
                for chunk in chain.stream(query):
                    if chunk and chunk.strip():
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append(AIMessage(content=full_response))
                
                # Display sources
                with st.expander("📚 Sources", expanded=False):
                    for idx, doc in enumerate(retrieved_docs[:3]):
                        st.markdown(f"""
                        **Source {idx + 1}:** {doc.metadata.get('source', 'Unknown')}  
                        **Section:** {doc.metadata.get('section', 'N/A')}  
                        **Page:** {doc.metadata.get('page', 'N/A')}
                        
                        ---
                        {doc.page_content[:300]}...
                        """)
                
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
                import traceback
                st.code(traceback.format_exc())


def build_rich_context(retrieved_docs):
    """Build context with metadata and structure"""
    context_parts = []
    
    for idx, doc in enumerate(retrieved_docs):
        metadata = doc.metadata
        
        context_part = f"""
[Document: {metadata.get('source', 'Unknown')}]
[Section: {metadata.get('section', 'N/A')}]
[Page: {metadata.get('page', 'N/A')}]
[Hierarchy Level: {metadata.get('hierarchy_level', 'N/A')}]

Content:
{doc.page_content}

---
"""
        context_parts.append(context_part)
    
    return "\n".join(context_parts)


def create_reasoning_prompt(show_reasoning=True):
    """Create prompt template with reasoning instructions"""
    if show_reasoning:
        template = """You are a legal document analysis expert. Analyze the following context and answer the question with structured reasoning.

Context from documents:
{context}

Question: {question}

Instructions:
1. Break down your reasoning into clear steps
2. Cite specific sections and page numbers
3. Highlight key entities (parties, dates, amounts)
4. Provide a clear, well-supported answer

Format your response as:

**Reasoning Steps:**
1. [First step of analysis]
2. [Second step]
...

**Answer:**
[Clear, comprehensive answer with citations]

**Key Entities:**
- Parties: [List]
- Dates: [List]
- Amounts: [List]
"""
    else:
        template = """Based on the following context, answer the question clearly and cite your sources.

Context:
{context}

Question: {question}

Provide a clear answer with specific citations (document name, section, page).
"""
    
    return PromptTemplate(template=template, input_variables=["context", "question"])


def render_document_comparison():
    """Render document comparison interface"""
    if len(st.session_state.documents) < 2:
        return
    
    with st.expander("📊 Document Comparison", expanded=False):
        st.subheader("Compare Documents")
        
        doc_names = list(st.session_state.documents.keys())
        
        col1, col2 = st.columns(2)
        
        with col1:
            doc1 = st.selectbox("Select Document 1", doc_names, key="doc1_select")
        
        with col2:
            doc2 = st.selectbox("Select Document 2", doc_names, key="doc2_select", index=1 if len(doc_names) > 1 else 0)
        
        comparison_type = st.radio(
            "Comparison Type",
            ["Structure", "Clauses", "Entities", "Full Analysis"]
        )
        
        if st.button("🔍 Compare Documents"):
            comparator = DocumentComparator()
            
            if comparison_type == "Structure":
                result = comparator.compare_structure(
                    st.session_state.documents[doc1],
                    st.session_state.documents[doc2]
                )
                st.json(result)
            
            elif comparison_type == "Clauses":
                result = comparator.compare_clauses(
                    st.session_state.documents[doc1],
                    st.session_state.documents[doc2]
                )
                
                # Display as table
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
            
            elif comparison_type == "Entities":
                result = comparator.compare_entities(
                    st.session_state.documents[doc1],
                    st.session_state.documents[doc2]
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{doc1} Entities:**")
                    st.json(result[doc1])
                with col2:
                    st.write(f"**{doc2} Entities:**")
                    st.json(result[doc2])


def main():
    """Main application flow"""
    init_session_state()
    configure_page()
    apply_custom_css()
    
    # Sidebar
    selected_model, api_key = handle_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Documents", 
        "💬 Chat", 
        "🕸️ Knowledge Graph",
        "📊 Analytics"
    ])
    
    with tab1:
        # Document upload and processing
        uploaded_files = render_document_upload_section()
        
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary"):
                process_documents(uploaded_files)
        
        # Document overview
        render_document_overview()
        
        # Document comparison
        render_document_comparison()
    
    with tab2:
        # Quick actions
        quick_query = render_quick_actions()
        
        # Chat interface
        chat_model = None
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            chat_model = get_chat_model(selected_model, api_key)
        
        display_chat_interface()
        
        if chat_model is None:
            st.warning("Please enter your API key in the sidebar")
        
        # Handle user input
        handle_user_query(chat_model, quick_query)
    
    with tab3:
        # Knowledge graph visualization
        render_knowledge_graph()
        
        # Entity statistics
        if st.session_state.get('knowledge_graph'):
            kg = st.session_state.knowledge_graph
            
            st.subheader("📈 Entity Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Entities", len(kg.nodes()))
            
            with col2:
                st.metric("Total Relationships", len(kg.edges()))
            
            with col3:
                entity_types = [kg.nodes[n].get('type', 'Unknown') for n in kg.nodes()]
                unique_types = len(set(entity_types))
                st.metric("Entity Types", unique_types)
    
    with tab4:
        st.subheader("📊 Document Analytics")
        
        if st.session_state.documents:
            # Aggregate statistics
            total_chunks = sum(len(doc['chunks']) for doc in st.session_state.documents.values())
            total_sections = sum(
                len(doc.get('structure', {}).get('sections', [])) 
                for doc in st.session_state.documents.values()
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Documents</div>
                    </div>
                    """.format(len(st.session_state.documents)),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Sections</div>
                    </div>
                    """.format(total_sections),
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Chunks</div>
                    </div>
                    """.format(total_chunks),
                    unsafe_allow_html=True
                )
            
            with col4:
                avg_chunks = total_chunks // len(st.session_state.documents) if st.session_state.documents else 0
                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Avg Chunks/Doc</div>
                    </div>
                    """.format(avg_chunks),
                    unsafe_allow_html=True
                )
        else:
            st.info("Upload and process documents to see analytics")


if __name__ == "__main__":
    main()
