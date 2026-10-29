import ollama
import streamlit as st

from src.generation import extract_citations_from_chunks, is_refusal_response
from src.pipeline import RAGPipeline

st.set_page_config(
    page_title="RBI Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RBI Document Assistant")
st.caption("Ask questions about Reserve Bank of India documents.")

@st.cache_resource
def load_pipeline():
    return RAGPipeline(use_reranker=True)

with st.spinner("Loading RAG pipeline and local models..."):
    pipeline = load_pipeline()

question = st.text_area(
    "Enter your question",
    placeholder="What is the purpose of the Cash Reserve Ratio (CRR)?",
    height=100,
)

top_k = st.slider(
    "Number of document chunks to retrieve",
    min_value=1,
    max_value=10,
    value=pipeline.config.retrieval.top_k,
)

if st.button("Ask RBI Assistant", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching and reranking RBI documents..."):
                stream, chunks = pipeline.answer_stream(question.strip(), top_k=top_k)

            st.subheader("Answer")
            answer_text = st.write_stream(stream)

            is_refusal = is_refusal_response(answer_text, pipeline.config.generation.refusal)
            citations = [] if is_refusal else extract_citations_from_chunks(chunks)

            if citations:
                st.subheader("Sources / Citations")
                for citation in citations:
                    st.markdown(f"- {citation}")

            if chunks:
                with st.expander("View retrieved document chunks"):
                    for i, chunk in enumerate(chunks, start=1):
                        st.markdown(
                            f"**{i}. {chunk.get('source_file', 'Unknown document')}**"
                        )
                        st.write(f"Pages: {chunk.get('pages', [])}")
                        st.text(chunk.get("text", ""))
                        st.divider()

            if is_refusal:
                st.info("The system could not find sufficient supporting information.")

        except (
            ollama.ResponseError,
            ollama.RequestError,
            ConnectionError,
            TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
        ) as exc:
            st.error(f"Unable to process the question: {exc}")
