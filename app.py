import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.title("📚 Local LangChain Pipline Bot")

uploaded_file = st.file_uploader("Upload a text file", type="txt")

if uploaded_file is not None:
    with open("uploaded.txt", "wb") as f:
        f.write(uploaded_file.read())
    loader = TextLoader("uploaded.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # ✅ Free embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # ✅ Free local model
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

    query = st.text_input("Ask a question about your file:")
    if query:
        # retrieve top documents
        docs = vectorstore.similarity_search(query, k=2)
        context = " ".join([d.page_content for d in docs])

        # generate answer
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        response = llm_pipeline(prompt, max_length=200, truncation=True)
        answer = response[0]['generated_text']

        st.success(answer)