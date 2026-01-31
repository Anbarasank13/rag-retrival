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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import custom modules
from document_processor_optimized import HierarchicalDocumentProcessor
from knowledge_graph_optimized import KnowledgeGraphBuilder
from retrieval_strategies_optimized import HybridRetriever
from clause_extractor import ClauseExtractor
from comparison_engine_optimized import DocumentComparator

# Constants - OPTIMIZED
CHUNK_SIZE = 800  # Reduced for better precision
CHUNK_OVERLAP = 150  # Optimized overlap
EMBEDDING_MODEL = "models/text-embedding-004"
RETRIEVER_K = 4  # Reduced for faster retrieval
MAX_CHUNKS_PER_DOC = 100  # Limit chunks per document
DEFAULT_SYSTEM_MESSAGE = """
You are an Advanced Hierarchical RAG Assistant for Legal Document Analysis 📄⚖️.

Provide clear, concise answers with proper citations. Always reference section numbers and page numbers.
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
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False


def configure_page():
    st.set_page_config(
        page_title="Hierarchical RAG - Legal Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("⚖️ Hierarchical RAG Legal Document Assistant")
    st.markdown("### Advanced Multi-Document Intelligence & Structured Reasoning")


def apply_custom_css():
    st.markdown(
        """
        <style>
        /* Performance optimizations */
        .stApp {
            max-width: 100%;
        }
        
        /* Improved document card */
        .doc-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            margin: 0.8rem 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            transition: transform 0.2s;
        }
        
        .doc-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            text-align: center;
            border-left: 4px solid #667eea;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Citation styling */
        .citation {
            background: #e3f2fd;
            padding: 0.8rem;
            border-left: 4px solid #2196f3;
            margin: 0.8rem 0;
            border-radius: 6px;
            font-size: 0.9em;
        }
        
        /* Tooltip styling */
        .tooltip-icon {
            color: #667eea;
            cursor: help;
            margin-left: 5px;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background-color: #667eea;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Comparison table */
        .comparison-row {
            padding: 1rem;
            margin: 0.5rem 0;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 3px solid #667eea;
        }
        
        /* Loading spinner */
        .stSpinner > div {
            border-color: #667eea !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_tooltip(text: str, tooltip: str):
    """Render text with tooltip"""
    return f'{text} <span class="tooltip-icon" title="{tooltip}">ⓘ</span>'


def handle_sidebar():
    """Enhanced sidebar with tooltips and better organization"""
    st.sidebar.header("🔑 Configuration")

    # API Key with tooltip
    st.sidebar.markdown(
        render_tooltip("**API Key**", "Your Google Gemini API key for AI processing"),
        unsafe_allow_html=True
    )
    
    api_key = st.sidebar.text_input(
        "Enter your Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        value=st.session_state.get("api_key", ""),
        label_visibility="collapsed"
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

    # Model Selection with tooltip
    st.sidebar.markdown(
        render_tooltip("**Generation Model**", "AI model used for answering questions. Flash is faster, Pro is more accurate."),
        unsafe_allow_html=True
    )
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        [
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state.model = selected_model

    st.sidebar.divider()

    # Retrieval Settings
    st.sidebar.subheader("⚙️ Retrieval Settings")
    
    st.sidebar.markdown(
        render_tooltip("**Retrieval Strategy**", 
        "Hybrid: Best balance (semantic + keyword). Dense: Pure semantic. Hierarchical: Section-aware search."),
        unsafe_allow_html=True
    )
    
    retrieval_mode = st.sidebar.radio(
        "Strategy",
        ["Hybrid (Best)", "Dense Only", "Hierarchical"],
        help="How to find relevant content in documents",
        label_visibility="collapsed"
    )
    st.session_state.retrieval_mode = retrieval_mode
    
    st.sidebar.markdown(
        render_tooltip("**Results Count**", "Number of document chunks to retrieve. Higher = more context but slower."),
        unsafe_allow_html=True
    )
    
    top_k = st.sidebar.slider("Top K", 2, 8, 4, label_visibility="collapsed")
    st.session_state.top_k = top_k

    st.sidebar.divider()

    # Feature Toggles
    st.sidebar.subheader("✨ Features")
    
    st.sidebar.markdown(
        render_tooltip("**Reasoning Steps**", "Show AI's step-by-step thinking process"),
        unsafe_allow_html=True
    )
    show_reasoning = st.sidebar.checkbox("Show Reasoning", value=True, label_visibility="collapsed")
    st.session_state.show_reasoning = show_reasoning
    
    st.sidebar.markdown(
        render_tooltip("**Knowledge Graph**", "Extract and visualize entities (people, companies, dates) and their relationships"),
        unsafe_allow_html=True
    )
    enable_kg = st.sidebar.checkbox("Build Knowledge Graph", value=True, label_visibility="collapsed")
    st.session_state.enable_kg = enable_kg

    st.sidebar.divider()

    # Document Stats
    st.sidebar.subheader("📚 Documents")
    
    num_docs = len(st.session_state.documents)
    
    if num_docs > 0:
        st.sidebar.success(f"✅ {num_docs} document(s) loaded")
        
        if st.sidebar.button("🗑️ Clear All", use_container_width=True, 
                            help="Remove all uploaded documents and reset the system"):
            st.session_state.documents = {}
            st.session_state.document_structures = {}
            st.session_state.knowledge_graph = None
            st.session_state.hybrid_retriever = None
            st.session_state.processing_complete = False
            st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
            st.rerun()
    else:
        st.sidebar.info("📄 No documents loaded")

    st.sidebar.divider()

    # Chat Controls
    if len(st.session_state.messages) > 1:
        if st.sidebar.button("🗑️ Clear Chat", use_container_width=True,
                            help="Clear conversation history"):
            st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
            st.rerun()
        
        # Export
        chat_text = ""
        for msg in st.session_state.messages[1:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            chat_text += f"{role}: {msg.content}\n\n"
        
        st.sidebar.download_button(
            "📥 Export Chat",
            chat_text,
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            use_container_width=True,
            help="Download conversation as text file"
        )

    return selected_model, st.session_state.get("api_key")


def render_document_upload():
    """Document upload with better UX"""
    st.markdown("### 📁 Upload Documents")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload legal documents for analysis. Supports PDF and TXT formats.",
            label_visibility="collapsed"
        )
    
    with col2:
        if uploaded_files:
            st.metric("Files Selected", len(uploaded_files))
    
    return uploaded_files


@st.cache_data(show_spinner=False, max_entries=5)
def process_single_document(_file_content, file_name, api_key):
    """Process a single document with caching - OPTIMIZED"""
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_name.split('.')[-1]}") as tmp:
        tmp.write(_file_content)
        tmp_path = tmp.name
    
    try:
        # Load
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
        
        documents = loader.load()
        
        # Limit pages for performance
        if len(documents) > 50:
            documents = documents[:50]
            st.warning(f"⚠️ {file_name}: Limited to first 50 pages for performance")
        
        # Process
        processor = HierarchicalDocumentProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        doc_data = processor.process_document(documents, file_name)
        
        # Limit chunks
        if len(doc_data['chunks']) > MAX_CHUNKS_PER_DOC:
            doc_data['chunks'] = doc_data['chunks'][:MAX_CHUNKS_PER_DOC]
        
        return doc_data
        
    finally:
        os.unlink(tmp_path)


def process_documents(uploaded_files):
    """Optimized document processing with progress tracking"""
    
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one document")
        return
    
    user_api_key = st.session_state.get("api_key", "")
    if not user_api_key:
        st.error("❌ Please enter your API key in the sidebar")
        return
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        total_files = len(uploaded_files)
        processed_docs = {}
        
        # Process each file
        for idx, uploaded_file in enumerate(uploaded_files):
            progress = idx / total_files
            status_text.text(f"📄 Processing {uploaded_file.name} ({idx + 1}/{total_files})...")
            progress_bar.progress(progress)
            
            # Use cached processing
            doc_data = process_single_document(
                uploaded_file.getvalue(),
                uploaded_file.name,
                user_api_key
            )
            
            processed_docs[uploaded_file.name] = doc_data
        
        st.session_state.documents = processed_docs
        
        # Build retrieval system
        status_text.text("🧠 Building retrieval system...")
        progress_bar.progress(0.8)
        
        all_chunks = []
        for doc_data in processed_docs.values():
            all_chunks.extend(doc_data['chunks'])
        
        # Limit total chunks
        if len(all_chunks) > 500:
            all_chunks = all_chunks[:500]
            st.warning("⚠️ Limited to 500 chunks for optimal performance")
        
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(all_chunks, embeddings)
        
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            documents=all_chunks,
            top_k=st.session_state.get('top_k', 4)
        )
        
        st.session_state.hybrid_retriever = hybrid_retriever
        
        # Build knowledge graph if enabled
        if st.session_state.get('enable_kg', True):
            status_text.text("🕸️ Building knowledge graph...")
            progress_bar.progress(0.9)
            
            kg_builder = KnowledgeGraphBuilder()
            knowledge_graph = kg_builder.build_from_documents(processed_docs)
            st.session_state.knowledge_graph = knowledge_graph
        
        # Initialize clause extractor
        st.session_state.clause_extractor = ClauseExtractor()
        st.session_state.processing_complete = True
        
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Successfully processed {total_files} document(s)!")
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def render_document_overview():
    """Display document cards with metrics"""
    if not st.session_state.documents:
        return
    
    st.markdown("### 📊 Document Overview")
    
    # Create metrics row
    cols = st.columns(4)
    
    total_sections = sum(
        len(doc.get('structure', {}).get('sections', [])) 
        for doc in st.session_state.documents.values()
    )
    total_chunks = sum(len(doc['chunks']) for doc in st.session_state.documents.values())
    total_pages = sum(
        doc.get('metadata', {}).get('total_pages', 0) 
        for doc in st.session_state.documents.values()
    )
    
    with cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.documents)}</div>
                <div class="metric-label">Documents</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_sections}</div>
                <div class="metric-label">Sections</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with cols[2]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_chunks}</div>
                <div class="metric-label">Chunks</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with cols[3]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_pages}</div>
                <div class="metric-label">Pages</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Document cards
    doc_cols = st.columns(min(len(st.session_state.documents), 3))
    
    for idx, (doc_name, doc_data) in enumerate(st.session_state.documents.items()):
        with doc_cols[idx % 3]:
            sections = len(doc_data.get('structure', {}).get('sections', []))
            chunks = len(doc_data['chunks'])
            pages = doc_data.get('metadata', {}).get('total_pages', 'N/A')
            
            st.markdown(
                f"""
                <div class="doc-card">
                    <h4 style="margin-top:0;">📄 {doc_name}</h4>
                    <p style="margin:5px 0;"><strong>Sections:</strong> {sections}</p>
                    <p style="margin:5px 0;"><strong>Chunks:</strong> {chunks}</p>
                    <p style="margin:5px 0;"><strong>Pages:</strong> {pages}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_quick_actions():
    """Quick action buttons with tooltips"""
    if not st.session_state.documents:
        return
    
    st.markdown("### ⚡ Quick Actions")
    st.markdown("Click any button to ask a pre-defined question")
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_query = None
    
    with col1:
        if st.button("🔍 Termination Clauses", use_container_width=True,
                    help="Find all termination clauses and their conditions"):
            quick_query = "Find and summarize all termination clauses with notice periods and conditions"
    
    with col2:
        if st.button("💰 Payment Terms", use_container_width=True,
                    help="Extract all payment amounts and schedules"):
            quick_query = "Extract all payment terms including amounts, due dates, and late fees"
    
    with col3:
        if st.button("📅 Key Dates", use_container_width=True,
                    help="List all important dates in the documents"):
            quick_query = "List all important dates: effective dates, deadlines, and expiration dates"
    
    with col4:
        if st.button("⚖️ Compare All", use_container_width=True,
                    help="Compare key terms across all documents"):
            quick_query = "Compare payment terms, termination conditions, and liabilities across all documents"
    
    return quick_query


def render_knowledge_graph_optimized():
    """Optimized knowledge graph with better visualization"""
    if not st.session_state.get('knowledge_graph'):
        st.info("💡 Enable 'Build Knowledge Graph' in sidebar and reprocess documents")
        return
    
    kg = st.session_state.knowledge_graph
    
    if not kg or len(kg.nodes()) == 0:
        st.info("No entities found. Try uploading documents with more structured content.")
        return
    
    # Entity statistics
    st.markdown("### 📈 Entity Statistics")
    
    entity_types = {}
    for node in kg.nodes():
        node_type = kg.nodes[node].get('type', 'Unknown')
        entity_types[node_type] = entity_types.get(node_type, 0) + 1
    
    # Display as metrics
    cols = st.columns(min(len(entity_types), 4))
    for idx, (ent_type, count) in enumerate(sorted(entity_types.items(), key=lambda x: x[1], reverse=True)):
        with cols[idx % 4]:
            st.metric(ent_type, count)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive graph with Plotly - OPTIMIZED
    st.markdown("### 🕸️ Entity Relationship Graph")
    
    # Limit nodes for better visualization
    max_nodes = 30
    if len(kg.nodes()) > max_nodes:
        st.warning(f"⚠️ Showing top {max_nodes} entities for clarity")
        # Get most connected nodes
        degree_dict = dict(kg.degree())
        top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        subgraph_nodes = [n[0] for n in top_nodes]
        kg_display = kg.subgraph(subgraph_nodes)
    else:
        kg_display = kg
    
    # Create Plotly network graph
    try:
        import networkx as nx
        
        # Get positions using spring layout
        pos = nx.spring_layout(kg_display, k=1, iterations=50)
        
        # Create edge traces
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in kg_display.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        # Create node traces
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=20,
                colorbar=dict(
                    thickness=15,
                    title='Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            ),
            textposition="top center",
            textfont=dict(size=10)
        )
        
        for node in kg_display.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            
            # Truncate long names
            display_name = node if len(node) <= 20 else node[:17] + "..."
            node_trace['text'] += tuple([display_name])
        
        # Color nodes by degree
        node_adjacencies = []
        node_text = []
        for node in kg_display.nodes():
            adjacencies = len(list(kg_display.neighbors(node)))
            node_adjacencies.append(adjacencies)
            node_info = f"{node}<br>Type: {kg_display.nodes[node].get('type', 'Unknown')}<br>Connections: {adjacencies}"
            node_text.append(node_info)
        
        node_trace.marker.color = node_adjacencies
        node_trace.hovertext = node_text
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering graph: {str(e)}")
        st.info("Try processing fewer documents or enable simpler visualization")


def display_chat_messages():
    """Display chat history"""
    for message in st.session_state.messages[1:]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)


@st.cache_resource()
def get_chat_model(model_name: str, api_key: str):
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.1)


def create_optimized_prompt(show_reasoning: bool):
    """Create optimized prompt template"""
    if show_reasoning:
        template = """Analyze the context and answer concisely with proper citations.

Context:
{context}

Question: {question}

Provide a structured answer:
1. Brief reasoning (2-3 steps)
2. Clear answer with citations [Doc: name, Section: X, Page: Y]
3. List key entities if relevant

Keep the response focused and under 300 words."""
    else:
        template = """Based on the context, answer the question with specific citations.

Context:
{context}

Question: {question}

Answer with citations [Doc, Section, Page]."""
    
    return PromptTemplate(template=template, input_variables=["context", "question"])


def handle_user_query(chat_model, query=None):
    """Optimized query handling"""
    
    if query is None:
        query = st.chat_input(
            "Ask about your documents...",
            disabled=(chat_model is None or not st.session_state.processing_complete)
        )
    
    if not query or not query.strip():
        return
    
    st.session_state.messages.append(HumanMessage(content=query))
    
    with st.chat_message("user"):
        st.write(query)
    
    if not st.session_state.get('hybrid_retriever'):
        with st.chat_message("assistant"):
            error_msg = "❌ Please process documents first"
            st.error(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))
        return
    
    with st.chat_message("assistant"):
        try:
            # Retrieve with timer
            retrieval_start = time.time()
            
            retriever = st.session_state.hybrid_retriever
            
            # Use optimized retrieval
            if st.session_state.get('retrieval_mode') == "Hybrid (Best)":
                retrieved_docs = retriever.hybrid_search(query)
            elif st.session_state.get('retrieval_mode') == "Hierarchical":
                retrieved_docs = retriever.hierarchical_search(query)
            else:
                retrieved_docs = retriever.dense_search(query)
            
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_docs:
                no_context_msg = "🤷‍♂️ No relevant information found"
                st.warning(no_context_msg)
                st.session_state.messages.append(AIMessage(content=no_context_msg))
                return
            
            # Build context - optimized
            context = build_context(retrieved_docs[:st.session_state.get('top_k', 4)])
            
            # Generate response
            prompt_template = create_optimized_prompt(
                st.session_state.get('show_reasoning', True)
            )
            
            chain = (
                RunnableParallel({
                    "context": RunnableLambda(lambda x: context),
                    "question": RunnablePassthrough()
                })
                | prompt_template
                | chat_model
                | StrOutputParser()
            )
            
            generation_start = time.time()
            message_placeholder = st.empty()
            full_response = ""
            
            for chunk in chain.stream(query):
                if chunk and chunk.strip():
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            generation_time = time.time() - generation_start
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append(AIMessage(content=full_response))
            
            # Show sources in expander
            with st.expander(f"📚 Sources ({len(retrieved_docs)} documents) • Retrieved in {retrieval_time:.2f}s"):
                for idx, doc in enumerate(retrieved_docs[:3]):
                    st.markdown(f"""
**Source {idx + 1}:** {doc.metadata.get('source', 'Unknown')}  
**Section:** {doc.metadata.get('section', 'N/A')} | **Page:** {doc.metadata.get('page', 'N/A')}

{doc.page_content[:200]}...
                    
---
                    """)
            
            st.rerun()
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))


def build_context(docs):
    """Build optimized context"""
    context_parts = []
    for doc in docs:
        meta = doc.metadata
        context_parts.append(
            f"[{meta.get('source', 'Unknown')} - {meta.get('section', 'N/A')} - Page {meta.get('page', 'N/A')}]\n{doc.page_content}\n"
        )
    return "\n---\n".join(context_parts)


def render_comparison_tab():
    """Enhanced document comparison with visualizations"""
    if len(st.session_state.documents) < 2:
        st.info("📊 Upload at least 2 documents to use comparison features")
        return
    
    st.markdown("### 📊 Document Comparison")
    
    doc_names = list(st.session_state.documents.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        doc1 = st.selectbox("Document 1", doc_names, key="comp_doc1")
    
    with col2:
        doc2_options = [d for d in doc_names if d != doc1]
        doc2 = st.selectbox("Document 2", doc2_options, key="comp_doc2")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    comparison_type = st.radio(
        "What to compare:",
        ["📋 Structure", "📄 Clauses", "👥 Entities", "📊 Full Analysis"],
        horizontal=True
    )
    
    if st.button("🔍 Run Comparison", type="primary", use_container_width=True):
        with st.spinner("Comparing documents..."):
            comparator = DocumentComparator()
            
            doc1_data = st.session_state.documents[doc1]
            doc2_data = st.session_state.documents[doc2]
            
            if "Structure" in comparison_type:
                result = comparator.compare_structure(doc1_data, doc2_data)
                
                # Visualize structure comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Common Sections", len(result['common_section_titles']))
                
                with col2:
                    st.metric(f"Unique to {doc1}", len(result['unique_to_doc1']))
                
                with col3:
                    st.metric(f"Unique to {doc2}", len(result['unique_to_doc2']))
                
                # Show details
                if result['common_section_titles']:
                    st.markdown("#### ✅ Common Sections")
                    for title in result['common_section_titles'][:10]:
                        st.markdown(f"- {title}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if result['unique_to_doc1']:
                        st.markdown(f"#### 📄 Unique to {doc1}")
                        for title in result['unique_to_doc1'][:5]:
                            st.markdown(f"- {title}")
                
                with col2:
                    if result['unique_to_doc2']:
                        st.markdown(f"#### 📄 Unique to {doc2}")
                        for title in result['unique_to_doc2'][:5]:
                            st.markdown(f"- {title}")
            
            elif "Clauses" in comparison_type:
                results = comparator.compare_clauses(doc1_data, doc2_data)
                
                # Display as table
                if results:
                    df_data = []
                    for comp in results:
                        df_data.append({
                            'Clause Type': comp['clause_type'].replace('_', ' ').title(),
                            doc1: comp['doc1_count'],
                            doc2: comp['doc2_count'],
                            'Difference': abs(comp['doc1_count'] - comp['doc2_count'])
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    fig = px.bar(
                        df, 
                        x='Clause Type', 
                        y=[doc1, doc2],
                        barmode='group',
                        title="Clause Count Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif "Entities" in comparison_type:
                result = comparator.compare_entities(doc1_data, doc2_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### 📄 {doc1} Entities")
                    for ent_type, entities in result.get(doc1, {}).items():
                        if entities:
                            st.markdown(f"**{ent_type}:** {', '.join(entities[:5])}")
                
                with col2:
                    st.markdown(f"#### 📄 {doc2} Entities")
                    for ent_type, entities in result.get(doc2, {}).items():
                        if entities:
                            st.markdown(f"**{ent_type}:** {', '.join(entities[:5])}")
            
            else:  # Full Analysis
                st.markdown("#### 📊 Comprehensive Comparison")
                
                # Similarity analysis
                similarity = comparator.compare_content_similarity(doc1_data, doc2_data, threshold=0.6)
                
                # Display similarity score
                similarity_score = (similarity['similar_chunks_count'] / 
                                  max(similarity['total_doc1_chunks'], similarity['total_doc2_chunks'])) * 100
                
                st.metric("Overall Similarity", f"{similarity_score:.1f}%")
                
                # Progress bar for similarity
                st.progress(similarity_score / 100)
                
                # Structure comparison
                struct_comp = comparator.compare_structure(doc1_data, doc2_data)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Common Sections", len(struct_comp['common_section_titles']))
                with col2:
                    st.metric(f"{doc1} Sections", struct_comp['doc1_sections'])
                with col3:
                    st.metric(f"{doc2} Sections", struct_comp['doc2_sections'])
                
                # Similar content sections
                if similarity['similar_chunks']:
                    st.markdown("#### 🔗 Similar Content")
                    for chunk in similarity['similar_chunks'][:3]:
                        st.markdown(f"""
**Similarity: {chunk['similarity']:.0%}**  
{doc1} Section: {chunk['doc1_section']}  
{doc2} Section: {chunk['doc2_section']}

---
                        """)


def main():
    """Main application"""
    init_session_state()
    configure_page()
    apply_custom_css()
    
    # Sidebar
    selected_model, api_key = handle_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Documents", 
        "💬 Chat", 
        "🕸️ Knowledge Graph",
        "📊 Comparison"
    ])
    
    with tab1:
        uploaded_files = render_document_upload()
        
        if uploaded_files and not st.session_state.processing_complete:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                process_documents(uploaded_files)
        
        if st.session_state.processing_complete:
            render_document_overview()
    
    with tab2:
        quick_query = render_quick_actions()
        
        chat_model = None
        if api_key and st.session_state.processing_complete:
            os.environ["GOOGLE_API_KEY"] = api_key
            chat_model = get_chat_model(selected_model, api_key)
        
        display_chat_messages()
        
        if not st.session_state.processing_complete:
            st.info("📄 Process documents in the Documents tab to start chatting")
        elif chat_model is None:
            st.warning("⚠️ Enter API key in sidebar to start")
        
        handle_user_query(chat_model, quick_query)
    
    with tab3:
        render_knowledge_graph_optimized()
    
    with tab4:
        render_comparison_tab()


if __name__ == "__main__":
    main()
