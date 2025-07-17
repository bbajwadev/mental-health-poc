# app.py

# ─── Always re-ingest & re-train ───────────────────────────────────────────────
from src.load_data   import main as run_ingest
from src.train_model import train as run_train_model

run_ingest()
run_train_model()

# ─── Now start the Streamlit app ───────────────────────────────────────────────
import streamlit as st
import sqlite3

from src.llm_client import get_advice_llm
from src.model_utils import predict_advice_type
from src.retrieval   import semantic_fetch as fetch_examples

st.set_page_config(page_title="Counselor Guidance POC")
st.title("🌱 Mental Health Counselor Guidance")

mode = st.sidebar.radio("Feature", ["LLM Advice", "Classify Response", "Search Examples"])

if mode == "LLM Advice":
    text = st.text_area("Patient notes", height=200)
    if st.button("Get Advice"):
        if not text.strip():
            st.error("Enter some notes first.")
        else:
            # Fetch & dedupe examples
            raw_examples = fetch_examples(text, k=5)
            seen = set(); examples = []
            for ctx, resp in raw_examples:
                norm = " ".join(ctx.split())
                if norm not in seen:
                    seen.add(norm)
                    examples.append((ctx, resp))
                if len(examples) >= 3:
                    break

            st.subheader("🔍 Similar Past Dialogues")
            if examples:
                for ctx, resp in examples:
                    st.markdown(f"**Patient:** {ctx}\n> **Counselor:** {resp}\n")
            else:
                st.info("No similar examples found.")

            with st.spinner("Thinking…"):
                advice = get_advice_llm(text)
            st.subheader("💡 AI-Generated Advice")
            st.write(advice)

elif mode == "Classify Response":
    resp_text = st.text_area("Paste a counselor response", height=200)
    if st.button("Classify"):
        if resp_text.strip():
            label = predict_advice_type(resp_text)
            st.write("Type:", label)
        else:
            st.error("Enter a response first.")

else:  # Search Examples
    query = st.text_input("Search term")
    if st.button("Search"):
        if not query.strip():
            st.error("Enter a search term.")
        else:
            conn = sqlite3.connect("data/processed/conversations.db")
            cur = conn.execute(
                "SELECT context, response FROM conversations "
                "WHERE context LIKE ? LIMIT 5",
                (f"%{query}%",)
            )
            results = cur.fetchall()
            conn.close()

            if results:
                for ctx, resp in results:
                    st.markdown(f"**Patient:** {ctx}\n> **Counselor:** {resp}\n")
            else:
                st.info("No matches found.")
