# src/llm_client.py
import os
from dotenv import load_dotenv
import openai
from src.retrieval import semantic_fetch as fetch_examples

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_advice_llm(patient_text: str) -> str:
    shots = fetch_examples(patient_text, k=3)

    few_shot_msgs = []
    for ctx, resp_text in shots:
        few_shot_msgs.append({"role":"user",      "content":f"A patient says: “{ctx}”"})
        few_shot_msgs.append({"role":"assistant", "content": resp_text})

    messages = [
        {"role":"system","content":"You’re an experienced mental-health counselor."},
        *few_shot_msgs,
        {
            "role":"user",
            "content":(
                f"A patient says:\n\"{patient_text}\"\n\n"
                "Based on the above examples, provide concise, empathetic guidance."
            )
        }
    ]

    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300
    )
    return resp.choices[0].message.content
