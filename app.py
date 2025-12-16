import streamlit as st
import pandas as pd
from openai import OpenAI
import time

# =========================
# CONFIG & SETUP
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
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
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
# LOAD CREDENTIALS FROM SECRETS
# =========================
try:
    NEO4J_URI = st.secrets["NEO4J_URI"]
    NEO4J_USER = st.secrets["NEO4J_USER"]
    NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
    OPENROUTER_KEY = st.secrets["openrouterKey"]
    HF_TOKEN = st.secrets["hfToken"]
except KeyError as e:
    st.error(f"❌ Missing secret: {e}")
    st.error("Please add all required secrets in Streamlit Cloud Settings → Secrets")
    st.info("""
    Required secrets:
    - NEO4J_URI
    - NEO4J_USER
    - NEO4J_PASSWORD
    - openrouterKey
    - hfToken
    """)
    st.stop()

# Import Backend
try:
    from fpl_Task2 import FPLGraphRetrieval
except ImportError:
    st.error("❌ Missing fpl_Task2.py. Please make sure the file exists.")
    st.stop()

# =========================
# CACHED RESOURCE LOADING
# =========================
@st.cache_resource(show_spinner=False)
def get_retriever_cached(uri, user, password, token):
    """
    Initializes the FPL Graph Retriever.
    Cached so model downloading (MiniLM/MPNet) happens only once.
    """
    print("🔄 Initializing Retriever (This happens once)...")
    return FPLGraphRetrieval(
        uri,
        user,
        password,
        hf_token=token,
        use_llm=True  # Enabled to show full integration capabilities
    )

def get_retriever():
    """Wrapper to handle loading UI feedback."""
    if "retriever" not in st.session_state or st.session_state.retriever is None:
        try:
            with st.spinner("🚀 Booting up System: Loading Embedding Models & Knowledge Graph... (Takes 30s first time)"):
                st.session_state.retriever = get_retriever_cached(
                    NEO4J_URI,
                    NEO4J_USER,
                    NEO4J_PASSWORD,
                    HF_TOKEN
                )
        except Exception as e:
            st.error(f"Failed to initialize retriever: {e}")
            st.stop()
    return st.session_state.retriever

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
        "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct:free",
        "Devstral 2512": "mistralai/devstral-2512:free",
        "Nemotron Nano 30B": "nvidia/nemotron-3-nano-30b-a3b:free"
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

    st.markdown("### 🔎 3. Retrieval Strategy")
    retrieval_mode = st.radio(
        "Search Method:",
        ["Baseline (Cypher Only)", "GraphRAG (Hybrid + Embeddings)"],
        index=1,
        help="Baseline: Exact database matches. GraphRAG: Combines exact matches with similar players."
    )
    
    method_param = "both" if "GraphRAG" in retrieval_mode else "baseline"
    
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
        - Use **GraphRAG** for finding similar players
        - Use **Baseline** for exact stats lookups
        - **Numeric embeddings** for stat-based similarity
        - **Text embeddings** for semantic understanding
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
st.caption(f"*Current Pipeline:* **{llm_choice}** → **{emb_choice}** → **{retrieval_mode}**")
st.markdown('</div>', unsafe_allow_html=True)

# Quick Stats Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("LLM Models", "3")
with col2:
    st.metric("Embedding Types", "3")
with col3:
    st.metric("Query Templates", "12+")
with col4:
    st.metric("Knowledge Graph", "Active")

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
            # 1. LOAD RETRIEVER
            retriever = get_retriever()
            
            # 2. SET EMBEDDING MODEL (Dynamic Switching)
            # Determine embedding type and set accordingly
            if selected_emb_type == "numeric":
                embedding_type_param = "numeric"
                status_container.write("✅ Using **Numeric Embeddings** (statistical similarity based on FPL stats)")
            elif selected_emb_type == "minilm":
                embedding_type_param = "minilm"
                retriever.active_model = 'model_1'
                status_container.write("✅ Using **MiniLM Text Embeddings** (384 dimensions, good balance)")
            elif selected_emb_type == "mpnet":
                embedding_type_param = "mpnet"
                retriever.active_model = 'model_2'
                status_container.write("✅ Using **MPNet Text Embeddings** (768 dimensions, most accurate)")
            
            # 3. PREPROCESSING & INTENT RECOGNITION
            status_container.write("🧠 Analyzing Intent & Extracting Entities...")
            preprocessed = retriever.preprocessor.preprocess(prompt, include_embedding=False)
            
            # Visualize detected entities
            intent = preprocessed.get('intent', 'unknown')
            entities = preprocessed.get('entities', {})
            
            # Show formatted metrics in the status
            col1, col2 = status_container.columns(2)
            with col1:
                st.metric("Detected Intent", intent.replace("_", " ").title())
                st.metric("Confidence", f"{preprocessed.get('intent_confidence', 0)*100:.0f}%")
            
            with col2:
                if entities:
                    st.write("📋 Extracted Entities:")
                    for key, values in entities.items():
                        if values:
                            st.caption(f"**{key}:** {', '.join(map(str, values[:3]))}")
            
            # 4. RETRIEVAL EXECUTION
            status_container.write(f"🔍 Retrieving Data using **{retrieval_mode}**...")
            result = retriever.retrieve(
                prompt, 
                method=method_param,
                embedding_type=embedding_type_param
            )
            
            # Extract data for visualization
            baseline_res = result.get('baseline', {}).get('results', [])
            embedding_res = result.get('embedding', {}).get('results', [])
            combined_res = result.get('combined', [])
            
            # Pick the best context source based on mode
            final_context = combined_res if method_param == "both" else baseline_res
            
            # 5. VISUALIZE RETRIEVED DATA
            if final_context:
                df_context = pd.DataFrame(final_context)
                status_container.write(f"📚 Retrieved **{len(final_context)}** relevant records:")
                
                # Show stats in a nice format
                st.markdown('<div class="fpl-stats">', unsafe_allow_html=True)
                st.dataframe(df_context.head(5), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show some key metrics if available
                if 'total_points' in df_context.columns:
                    top_player = df_context.iloc[0] if len(df_context) > 0 else None
                    if top_player is not None:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Top Points", f"{top_player.get('total_points', 0):.0f}")
                        with col2:
                            st.metric("Top Goals", f"{top_player.get('goals', 0):.0f}")
                        with col3:
                            st.metric("Top Assists", f"{top_player.get('assists', 0):.0f}")
            else:
                status_container.warning("⚠ No direct data found in Knowledge Graph.")
            
            # Show retrieval method breakdown
            if method_param == "both":
                status_container.write(f"📊 **Retrieval Breakdown:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Baseline Results", len(baseline_res))
                with col2:
                    st.metric("Embedding Results", len(embedding_res))
                with col3:
                    st.metric("Combined", len(combined_res))
            
            # Show Cypher query if available
            cypher_query = result.get('baseline', {}).get('cypher', None)
            if cypher_query:
                with st.expander("🔍 View Cypher Query"):
                    st.code(cypher_query, language="cypher")
            
            status_container.update(label="✅ Pipeline Complete", state="complete", expanded=False)

            # 6. GENERATION (LLM)
            context_str = str(final_context)[:4000]  # Truncate to avoid context limit
            
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_KEY
            )

            # System prompt with FPL expert persona
            system_prompt = f"""You are an expert Fantasy Premier League (FPL) assistant. 
You have access to FPL statistics and data. Use the provided context to answer the user's question accurately.

CONTEXT DATA:
{context_str}

IMPORTANT GUIDELINES:
1. Always provide a response - never return empty text
2. Cite specific stats from the context when available
3. For comparisons, create clear tables or bullet points
4. For recommendations, explain your reasoning
5. Keep responses concise but informative
6. Use FPL terminology appropriately

USER QUESTION:
{prompt}

ANSWER (remember to always provide a helpful response):"""

            # Stream response
            stream = client.chat.completions.create(
                model=selected_llm_id,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                stream=True,
                temperature=0.3,
                max_tokens=800
            )
            
            response = st.write_stream(stream)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            status_container.update(label="❌ Error", state="error")
            st.error(f"An error occurred: {e}")
            st.info("Try simplifying your query or checking your configuration.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("⚽ **FPL Graph-RAG System**")
with col2:
    st.caption("📊 Powered by Neo4j Knowledge Graph")
with col3:
    st.caption("🤖 Enhanced with LLM + Embeddings")
