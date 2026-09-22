import os
import streamlit as st
from src.rag import answer_question

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="📚", layout="centered")

st.title("📚 Domain-Specific RAG Chatbot")
st.caption("Ask questions from the provided document corpus. Answers are grounded in retrieved document chunks.")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY is not set. Add it to your environment or Streamlit secrets before asking a question.")

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What is the minimum attendance requirement?"
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving relevant passages and generating an answer..."):
        try:
            result = answer_question(question.strip())
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            for source in result["sources"]:
                st.markdown(
                    f"**{source['file']} — Page {source['page']}**  \n"
                    f"{source['preview']}"
                )
        except Exception as exc:
            st.error(f"Error: {exc}")

st.divider()
st.caption("The bot refuses questions when the answer is not supported by the document corpus.")
