import os, requests, streamlit as st

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def query(payload):
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
    return r.json()[0]["generated_text"]

st.title("6年生向けAI検索（API版）")
question = st.text_input("質問を入力してね")
if st.button("やさしい答え") and question:
    with st.spinner("考え中…"):
        prompt = f"小学生にも分かる日本語で50字以内で説明:\n質問: {question}\n答え:"
        ans = query({"inputs": prompt, "options": {"wait_for_model": True}})
    st.success(ans.strip())
