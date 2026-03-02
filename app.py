import streamlit as st
import rag_engine as rag
import os

st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

def main():
    st.title("🧠 DocuMind AI: Enterprise RAG Analyst")
    st.markdown("Upload your PDFs, let the system process them, and chat with your data!")

    # Initialize session state
    if "is_processed" not in st.session_state:
        st.session_state.is_processed = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Sidebar ---
    with st.sidebar:
        st.title("📁 Document Management")
        st.markdown("Upload your PDF Files and click 'Process'")
        
        pdf_docs = st.file_uploader(
            "Drag and drop files here", 
            accept_multiple_files=True, 
            type=["pdf"]
        )
        
        if st.button("Process Documents"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file first.")
            else:
                with st.spinner("Processing documents..."):
                    raw_text = rag.get_pdf_text(pdf_docs)
                    text_chunks = rag.get_text_chunks(raw_text)
                    rag.get_vector_store(text_chunks)
                    
                    st.session_state.is_processed = True
                    st.success("Documents processed successfully! You can now chat.")

    # --- Main Chat Interface ---
    if not st.session_state.is_processed:
        st.info("👋 Please upload and process a PDF document in the sidebar to start chatting.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your documents...", disabled=not st.session_state.is_processed)

    if prompt and st.session_state.is_processed:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing documents..."):
            response = rag.process_user_query(prompt)
            
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()