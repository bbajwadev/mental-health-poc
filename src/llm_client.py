# src/llm_client.py

import os
from dotenv import load_dotenv
import openai

# Load .env in local development
load_dotenv()

# Attempt to get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# If running inside Streamlit, try secrets as fallback
try:
    import streamlit as st
    # Use secret if API key not set in env
    secret_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key and secret_key:
        api_key = secret_key
except Exception:
    pass

if not api_key:
    raise ValueError(
        "OpenAI API key not found. Set OPENAI_API_KEY in .env or Streamlit secrets."
    )

openai.api_key = api_key

from src.retrieval import semantic_fetch as fetch_examples

def get_advice_llm(patient_text: str) -> str:
    """
    Retrieve semantically similar past dialogues, use them as few-shot examples,
    then call the OpenAI chat-completions API for guidance.
    """
    # 1) Fetch semantic examples
    shots = fetch_examples(patient_text, k=3)

    # 2) Build few-shot message list
    few_shot_msgs = []
    for ctx, resp_text in shots:
        few_shot_msgs.append({
            "role": "user",
            "content": f"A patient says: “{ctx}”"
        })
        few_shot_msgs.append({
            "role": "assistant",
            "content": resp_text
        })

    # 3) Construct final prompt messages
    messages = [
        {"role": "system", "content": "You’re an experienced mental-health counselor."},
        *few_shot_msgs,
        {
            "role": "user",
            "content": (
                f"A patient says:\n\"{patient_text}\"\n\n"
                "Based on the above examples, provide concise, empathetic guidance."
            )
        }
    ]

    # 4) Call OpenAI chat-completions
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300
    )

    return resp.choices[0].message.content
