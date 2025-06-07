import streamlit as st
import re
import subprocess
import os

from langchain.chains.retrieval_qa import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import shutil

# Configuration
VECTOR_DB_PATH = "faiss_index"
DEEPSEEK_API_KEY = "sk-487936a049ba4e65a7c7a43101efc099"

# STEP 1: Load Automation Code Files
loader = DirectoryLoader(
    path="C:\\Users\\chingepallykumar\\PycharmProjects\\LLMDemo",
    glob="**/*.py",
    loader_cls=TextLoader
)

# Loader for .feature files
feature_loader = DirectoryLoader(
    path="C:\\Users\\chingepallykumar\\PycharmProjects\\LLMDemo",
    glob="**/*.feature",
    loader_cls=TextLoader
)

# Load documents
documents = loader.load() + feature_loader.load()

embedding_model = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-xl"
)

embedding_model = HuggingFaceEmbeddings()
if os.path.exists(VECTOR_DB_PATH):
    print("✅ Loading existing vector DB...")
    vectordb = FAISS.load_local(VECTOR_DB_PATH, embedding_model,allow_dangerous_deserialization=True)
else:
    print("🚀 Building vector DB for the first time...")
    # Load and process documents (as you do)    
    splitter = PythonCodeTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    vectordb = FAISS.from_documents(docs, embedding_model)
    vectordb.save_local(VECTOR_DB_PATH)
    
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm = ChatOpenAI(
    model_name="deepseek-chat",
    temperature=0,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com/v1"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Function to extract test functions or file paths
def extract_test_targets(source_docs):
    test_targets = set()
    for doc in source_docs:
        file_path = doc.metadata["source"]
        code = doc.page_content
        match = re.search(r"def (test_\w+)\(", code)
        if match:
            test_func = match.group(1)
            target = f"{file_path}::{test_func}"
        else:
            target = file_path
        test_targets.add(target)
    return list(test_targets)

# Function to run tests and return output
def run_pytest_with_output(target):
    try:
        result = subprocess.run(
            ["pytest", target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout + ("\n⚠️ " + result.stderr if result.stderr.strip() else "")
    except Exception as e:
        return f"❌ Error: {e}"

# Streamlit UI
st.title("🧠 RAG-Powered Pytest Assistant")

if "source_docs" not in st.session_state:
    st.session_state.source_docs = None

query = st.text_input("Ask a question about your automation code:")

if st.button("Submit Query") and query.strip():
    with st.spinner("Thinking..."):
        result = qa_chain(query)
        st.success(result["result"])
        st.session_state.source_docs = result["source_documents"]

# Show source document checkboxes only if available
if st.session_state.source_docs:
    st.markdown("### 📂 Retrieved test functions")
    selected_indices = []
    st.markdown("#### 📂 Select the tests if you want to run.")
    for i, doc in enumerate(st.session_state.source_docs):
        if st.checkbox(f"[{i}] {doc.metadata['source']}", key=f"chk_{i}"):
            selected_indices.append(i)

    if selected_indices:
        if st.button("🧪 Run Selected Tests"):
            selected_docs = [st.session_state.source_docs[i] for i in selected_indices]
            test_targets = extract_test_targets(selected_docs)

            st.markdown("### 🔬 Test Targets")
            st.code("\n".join(test_targets))

            for target in test_targets:
                st.markdown(f"**▶ Running:** `{target}`")
                output = run_pytest_with_output(target)
                st.code(output, language="bash")
