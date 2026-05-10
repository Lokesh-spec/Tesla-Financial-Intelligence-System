import os
import streamlit as st
from query_engine import load_config, extract_years, build_enhanced_query, retrieve_chunks, deduplicate_chunks, classify_query
from generation.generator import generate_answer

CONFIG_PATH = "config/parameters.yaml"

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ── Page config ──
st.set_page_config(
    page_title="Tesla Financial Intelligence System",
    page_icon="🚗",
    layout="wide"
)

# ── Load config and vector db once ──
@st.cache_resource
def load_resources():
    from retrieval.retriever import load_vector_db
    config = load_config(CONFIG_PATH)
    vectordb = load_vector_db(config)
    return config, vectordb

config, vectordb = load_resources()

# ── Sidebar ──
with st.sidebar:
    st.image("images/RAG_Pipeline.png", use_container_width=True)
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This system allows you to query **7 years of Tesla 10-K filings** "
        "(2017–2024) using natural language."
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    - 📄 **18,379 vectors** stored in ChromaDB
    - 🔍 **Dynamic Top-K** retrieval based on query type
    - 🧠 **GPT** answers strictly from retrieved context
    - 📅 **Metadata filtering** by year
    """)
    st.markdown("---")
    st.markdown("### Query Types")
    st.markdown("""
    | Type | Keywords | Top-K |
    |---|---|---|
    | Specific | default | 5 |
    | Summary | overview, explain | 10 |
    | Comparison | compare, vs | 15 |
    """)

    # Session info
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        st.markdown("---")
        st.markdown("### Session Info")
        total = len([m for m in st.session_state["messages"] if m["role"] == "user"])
        st.markdown(f"**Questions asked:** {total}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ── Main UI ──
st.title("🚗 Tesla Financial Intelligence System")
st.markdown("Ask questions about Tesla's financial performance from **2017 to 2024**.")
st.markdown("---")

# ── Initialise session state ──
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── Display chat history ──
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            with st.expander("View source chunks"):
                for m in message["metadata"]:
                    st.write(m)

# ── Chat input ──
query = st.chat_input("Ask a question about Tesla's financials...")

if query:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Step 1 — classify
            search_params = classify_query(query)
            top_k = search_params["k"]
            query_type = search_params["type"]

            # Step 2 — extract years
            years = extract_years(query)

            # Step 3 — enhance query
            enhanced_query = build_enhanced_query(query, years)

            # Step 4 — retrieve
            raw_docs = retrieve_chunks(vectordb, enhanced_query, years, top_k)
            docs = deduplicate_chunks(raw_docs)

            # Step 5 — generate
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = generate_answer(context, query, query_type)

            # Step 6 — display
            st.markdown(answer)

            # Step 7 — show metadata
            sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
            col1, col2, col3 = st.columns(3)
            col1.metric("Query Type", query_type.capitalize())
            col2.metric("Chunks Retrieved", len(docs))
            col3.metric("Years Detected", ", ".join(years) if years else "All")

            with st.expander("📄 View source files used"):
                for s in sources:
                    st.write(f"• {s}")

    # Save to history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "metadata": [doc.metadata for doc in docs]
    })