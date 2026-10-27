import streamlit as st

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
    value=5,
)

if st.button("Ask RBI Assistant", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching RBI documents and generating an answer..."):
            try:
                result = pipeline.answer(question.strip(), top_k=top_k)

                st.subheader("Answer")
                st.write(result["answer"])

                citations = result.get("citations", [])
                if citations:
                    st.subheader("Sources / Citations")
                    for citation in citations:
                        st.markdown(f"- {citation}")

                chunks = result.get("retrieved_chunks", [])
                if chunks:
                    with st.expander("View retrieved document chunks"):
                        for i, chunk in enumerate(chunks, start=1):
                            st.markdown(
                                f"**{i}. {chunk.get('source_file', 'Unknown document')}**"
                            )
                            st.write(f"Pages: {chunk.get('pages', [])}")
                            st.text(chunk.get("text", ""))
                            st.divider()

                if result.get("is_refusal"):
                    st.info("The system could not find sufficient supporting information.")

            except Exception as exc:
                st.error(f"Unable to process the question: {exc}")
