import re, streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource(show_spinner="モデルをロード中…⌛")
def load_pipe():
    model_id = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)

pipe = load_pipe()
CACHE = {}

st.title("6年生向けAI検索")
st.write("どんな質問でも小学6年生にわかる言葉で答えます。")

q = st.text_input("質問を入力してね")

if st.button("答えを見る") and q:
    if q in CACHE:
        st.success(CACHE[q])
    else:
        prompt = f"小学生にもわかる日本語で50字以内で説明: {q}"
        txt = pipe(prompt, max_new_tokens=64, num_beams=4, do_sample=False)[0]["generated_text"]
        ans = re.sub(r"\s+", " ", txt).strip()
        CACHE[q] = ans
        st.success(ans)
