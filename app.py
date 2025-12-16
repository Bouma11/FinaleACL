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

# Load Config
def load_config():
    conf = {}
    try:
        with open("config.txt", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    conf[k] = v
    except FileNotFoundError:
        pass
    return conf

config = load_config()

# Credentials
NEO4J_URI = config.get("URI", "")
NEO4J_USER = config.get("USERNAME", "")
NEO4J_PASSWORD = config.get("PASSWORD", "")
OPENROUTER_KEY = config.get("OPENROUTER_KEY", "")
HF_TOKEN = config.get("HF_TOKEN", "")

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
    st.title("⚙ System Control")
    
    st.markdown("### 🧠 1. Generation Model")
    st.caption("Select the LLM that generates the final answer.")
    llm_map = {
        "Llama 3 70B": "meta-llama/llama-3.3-70b-instruct:free",
        "Mistral 7B": "mistralai/mistral-7b-instruct:free",
        "Gemini Flash": "google/gemini-2.0-flash-exp:free"
    }
    llm_choice = st.selectbox("LLM Model", list(llm_map.keys()), index=0)
    selected_llm_id = llm_map[llm_choice]

    st.markdown("---")

    st.markdown("### 📐 2. Embedding Model")
    st.caption("Select the model used for semantic search integration.")
    # Mapping UI names to the keys used in fpl_Task2.py
    embedding_map = {
        "MiniLM (Fast, 384d)": "model_1",
        "MPNet (Accurate, 768d)": "model_2"
    }
    emb_choice = st.selectbox("Embedding Model", list(embedding_map.keys()), index=0)
    selected_emb_id = embedding_map[emb_choice]

    st.markdown("---")

    st.markdown("### 🔎 3. Retrieval Strategy")
    retrieval_mode = st.radio(
        "Method:",
        ["Baseline (Cypher Only)", "GraphRAG (Hybrid + Embeddings)"],
        index=1,
        help="Baseline uses exact matches. GraphRAG combines Knowledge Graph with Vector Similarity."
    )
    
    # Map selection to method string expected by retrieve()
    method_param = "both" if "GraphRAG" in retrieval_mode else "baseline"

    if st.button("🗑 Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# =========================
# MAIN INTERFACE
# =========================
st.title("⚽ FPL Graph-RAG System")
st.caption(f"*Current Pipeline:* {llm_choice} ➡ {emb_choice} ➡ {retrieval_mode}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input handling
if prompt := st.chat_input("Ask about FPL players, teams, or stats..."):
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
            # This fulfills the requirement to switch embedding models
            retriever.active_model = selected_emb_id 
            status_container.write(f"✅ Active Embedding Model set to: *{emb_choice}*")
            
            # 3. PREPROCESSING & INTENT RECOGNITION
            status_container.write("🧠 Analyzing Intent & Extracting Entities...")
            # We call preprocess separately just to show it in the UI (Integration visibility)
            preprocessed = retriever.preprocessor.preprocess(prompt, include_embedding=False)
            
            # Visualize detected entities
            intent = preprocessed.get('intent', 'unknown')
            entities = preprocessed.get('entities', {})
            
            # Show formatted metrics in the status
            col1, col2 = status_container.columns(2)
            col1.metric("Detected Intent", intent.replace("_", " ").title())
            col2.json(entities) # Show extracted entities raw
            
            # 4. RETRIEVAL EXECUTION
            status_container.write(f"🔍 Retrieving Data using *{retrieval_mode}*...")
            result = retriever.retrieve(prompt, method=method_param)
            
            # Extract data for visualization
            baseline_res = result.get('baseline', {}).get('results', [])
            embedding_res = result.get('embedding', {}).get('results', [])
            combined_res = result.get('combined', [])
            
            # Pick the best context source based on mode
            final_context = combined_res if method_param == "both" else baseline_res
            
            # 5. VISUALIZE RETRIEVED DATA (The "Integration" part)
            if final_context:
                df_context = pd.DataFrame(final_context)
                status_container.write(f"📚 Retrieved {len(final_context)} relevant records:")
                status_container.dataframe(df_context.head(3), use_container_width=True)
            else:
                status_container.warning("⚠ No direct data found in Knowledge Graph.")

            # Show Cypher query if available (Demonstrates internal process)
            cypher_query = result.get('baseline', {}).get('cypher', None)
            if cypher_query:
                status_container.code(cypher_query, language="cypher")
                
            status_container.update(label="✅ Pipeline Complete", state="complete", expanded=False)

            # 6. GENERATION (LLM)
            context_str = str(final_context)[:6000] # Truncate to avoid context limit
            
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_KEY
            )

            # Stream response
            stream = client.chat.completions.create(
                model=selected_llm_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert FPL Assistant. "
                            "Use the provided Knowledge Graph context to answer the user's question accurately. "
                            "If the context contains stats, cite them explicitly."
                        )
                    },
                    {
                        "role": "user", 
                        "content": f"CONTEXT DATA:\n{context_str}\n\nUSER QUESTION:\n{prompt}"
                    }
                ],
                stream=True
            )
            
            response = st.write_stream(stream)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            status_container.update(label="❌ Error", state="error")
            st.error(f"An error occurred: {e}")