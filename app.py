"""
FPL Graph-RAG Streamlit Frontend
This app uses Task3.py's call_llm function to process user queries
"""

import streamlit as st
import pandas as pd
from Task3 import call_llm, retriever

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="FPL Graph-RAG Assistant",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS for FPL theme
st.markdown("""
    <style>
    .manager-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fpl-stats {
        background-color: #f8f9fa;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton button {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #FF8E53 0%, #FF6B6B 100%);
        color: white;
    }
    .sidebar-header {
        background: linear-gradient(90deg, #2c3e50 0%, #4ca1af 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.title("⚙ FPL Manager Dashboard")
    st.caption("Configure your AI Assistant")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🧠 1. Generation Model")
    st.caption("Select the LLM that generates the final answer.")
    llm_map = {
        "Mistral 7B": "mistralai/mistral-7b-instruct:free",
        "Llama 3 70B": "meta-llama/llama-3.3-70b-instruct:free",
        "Devstral": "mistralai/devstral-2512:free",
        "Nemotron 3": "nvidia/nemotron-3-nano-30b-a3b:free"
    }
    llm_choice = st.selectbox("LLM Model", list(llm_map.keys()), index=0)
    selected_llm_id = llm_map[llm_choice]

    st.markdown("---")

    st.markdown("### 📐 2. Embedding Model")
    st.caption("How the system finds similar players/stats.")
    embedding_map = {
        "Numeric (Stats-based)": "numeric",
        "MiniLM (Fast, 384d)": "minilm",
        "MPNet (Accurate, 768d)": "mpnet"
    }
    emb_choice = st.selectbox("Embedding Model", list(embedding_map.keys()), index=0)
    selected_emb_type = embedding_map[emb_choice]

    st.markdown("---")
    
    # FPL Tips Section
    with st.expander("💡 FPL Assistant Tips"):
        st.markdown("""
        **Best Query Examples:**
        - "Compare Haaland vs Salah"
        - "Top defenders for clean sheets"
        - "Which midfielders have most assists?"
        - "Erling Haaland gameweek 20 performance"
        - "Manchester City upcoming fixtures"
        
        **Pro Tips:**
        - Use **Numeric embeddings** for stat-based similarity
        - Use **Text embeddings** for semantic understanding
        """)
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LLM", llm_choice.split()[0])
    with col2:
        st.metric("Embeddings", selected_emb_type[:6])
    
    if st.button("🔄 Reset Conversation", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =========================
# MAIN INTERFACE
# =========================
st.markdown('<div class="manager-header">', unsafe_allow_html=True)
st.title("⚽ FPL Graph-RAG System")
st.markdown("### 👋 Welcome, Manager! Your AI Assistant for Fantasy Premier League Decisions")
st.caption(f"*Current Pipeline:* **{llm_choice}** → **{emb_choice}**")
st.markdown('</div>', unsafe_allow_html=True)

# Quick Stats Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("LLM Models", "4")
with col2:
    st.metric("Embedding Types", "3")
with col3:
    st.metric("Knowledge Graph", "Active")
with col4:
    st.metric("Retrieval Method", "GraphRAG")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello Manager! I'm your FPL AI Assistant. Ask me about players, teams, stats, or comparisons! ⚽"}
    ]

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input handling
if prompt := st.chat_input("Ask about FPL players, teams, or stats... e.g., 'Compare Haaland and Salah'"):
    
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant processing
    with st.chat_message("assistant"):
        # Create a container for the "Working..." status
        status_container = st.status("🏗 Processing Pipeline...", expanded=True)
        
        try:
            status_container.write(f"🤖 Using Model: **{llm_choice}**")
            status_container.write(f"📐 Using Embeddings: **{emb_choice}**")
            status_container.write("🔍 Retrieving data from Knowledge Graph...")
            
            # Call the LLM function from Task3.py
            answer, tokens_used = call_llm(
                user_question=prompt,
                model=selected_llm_id,
                embedding_type=selected_emb_type
            )
            
            status_container.write(f"✅ Generated response ({tokens_used} tokens used)")
            status_container.update(label="✅ Pipeline Complete", state="complete", expanded=False)
            
            # Display the answer
            st.markdown(answer)
            
            # Show token usage
            with st.expander("📊 Token Usage"):
                st.metric("Total Tokens", tokens_used)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            status_container.update(label="❌ Error", state="error")
            error_msg = f"An error occurred: {e}"
            st.error(error_msg)
            st.info("Try simplifying your query or checking your configuration in config.txt")
            
            # Save error to history
            st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("⚽ **FPL Graph-RAG System**")
with col2:
    st.caption("📊 Powered by Neo4j Knowledge Graph")
with col3:
    st.caption("🤖 Enhanced with LLM + Embeddings")
