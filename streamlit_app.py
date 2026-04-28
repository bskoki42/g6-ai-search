import pathlib
import requests
import streamlit as st

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
SLIDE_PATH = pathlib.Path("slides/codex-vs-gpt-ja.md")


def query(payload):
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
    return r.json()[0]["generated_text"]


def render_ai_search():
    st.title("6年生向けAI検索（API版）")
    question = st.text_input("質問を入力してね")
    if st.button("やさしい答え") and question:
        with st.spinner("考え中…"):
            prompt = f"小学生にも分かる日本語で50字以内で説明:\n質問: {question}\n答え:"
            ans = query({"inputs": prompt, "options": {"wait_for_model": True}})
        st.success(ans.strip())


def render_slides():
    st.title("Codex と GPT の違いスライド")
    st.caption("この画面でスライド原稿をそのまま確認できます。")

    if not SLIDE_PATH.exists():
        st.error(f"スライドファイルが見つかりません: {SLIDE_PATH}")
        return

    slide_text = SLIDE_PATH.read_text(encoding="utf-8")

    st.download_button(
        "スライドMarkdownをダウンロード",
        data=slide_text,
        file_name=SLIDE_PATH.name,
        mime="text/markdown",
    )

    st.subheader("プレビュー")
    for section in slide_text.split("\n---\n"):
        st.markdown(section)
        st.divider()

    with st.expander("外部ツールでスライド表示する方法（Marp）"):
        st.code(
            """npm i -g @marp-team/marp-cli
marp slides/codex-vs-gpt-ja.md --html --allow-local-files
# 生成されたHTMLをブラウザで開く""",
            language="bash",
        )


mode = st.sidebar.radio("モード", ["AI検索", "スライド閲覧"], index=1)

if mode == "AI検索":
    render_ai_search()
else:
    render_slides()
