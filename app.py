import streamlit as st
import torch
import os
import pandas as pd
import io
import sqlite3
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


# Page config
st.set_page_config(
    page_title="📄📊🗄️ Gemma Universal Data Chat",
    page_icon="🤖",
    layout="wide"
)

# Session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None


@st.cache_resource
def load_model():
    """Load Gemma model with HuggingFace pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner("Loading Gemma model..."):
        try:
            gemma_pipe = pipeline(
                "text-generation",
                model="google/gemma-2-2b-it",
                device_map="auto" if device.type == "cuda" else None,
                max_new_tokens=300,
                min_new_tokens=20,
                temperature=0.8,
                do_sample=True,
                return_full_text=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                eos_token_id=None,
                pad_token_id=50256
            )

            gemma_llm = HuggingFacePipeline(pipeline=gemma_pipe)

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="human_input"
            )

            prompt_template = PromptTemplate(
                input_variables=["chat_history", "human_input", "context"],
                template="""<bos>Previous conversation:
{chat_history}

Data Context:
{context}

<start_of_turn>user
{human_input}<end_of_turn>
<start_of_turn>model
"""
            )

            conversation = LLMChain(
                llm=gemma_llm,
                prompt=prompt_template,
                memory=memory,
                verbose=False
            )

            return conversation, device

        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None, device


def create_table_context(df, file_name):
    """Build Repomix-style context summary for tabular data"""
    context = []
    context.append(f"📂 File: {file_name}")
    context.append(f"🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    context.append("\n=== Column Analysis ===")

    for col in df.columns:
        col_data = df[col]
        summary = [f"\n📊 Column: {col}"]
        summary.append(f" - Data type: {col_data.dtype}")

        missing = col_data.isnull().sum()
        if missing > 0:
            summary.append(f" - Missing values: {missing} ({missing/len(col_data)*100:.2f}%)")

        unique_vals = col_data.nunique()
        summary.append(f" - Unique values: {unique_vals}")

        if pd.api.types.is_numeric_dtype(col_data):
            summary.append(f" - Min: {col_data.min()}, Max: {col_data.max()}, Mean: {col_data.mean():.2f}")
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            top_vals = col_data.value_counts().head(5).to_dict()
            summary.append(f" - Top categories: {top_vals}")

        context.append("\n".join(summary))

    context.append("\n=== Sample Data ===")
    context.append(df.head(5).to_string())

    return "\n".join(context)


def process_file(uploaded_file):
    """Process PDF or any tabular data file (CSV, Excel, SQL)"""
    try:
        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("❌ File too large! Please use a smaller file (<20MB).")
            return None

        file_name = uploaded_file.name.lower()

        # === PDF ===
        if file_name.endswith(".pdf"):
            with st.spinner("Processing PDF..."):
                temp_path = "temp.pdf"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                docs = PyPDFLoader(temp_path).load()
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=400, chunk_overlap=50
                ).split_documents(docs)

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(chunks, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

                os.remove(temp_path)
                return retriever

        # === CSV ===
        elif file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            context = create_table_context(df, file_name)

        # === Excel ===
        elif file_name.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
            df = pd.read_excel(uploaded_file, engine="openpyxl" if not file_name.endswith(".xlsb") else "pyxlsb")
            context = create_table_context(df, file_name)

        # === SQL (SQLite dump example) ===
        elif file_name.endswith(".sql"):
            conn = sqlite3.connect(":memory:")
            sql_script = uploaded_file.read().decode("utf-8")
            conn.executescript(sql_script)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)

            if tables.empty:
                return None

            table_name = tables.iloc[0, 0]
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()
            context = create_table_context(df, file_name)

        else:
            st.error(f"❌ Unsupported file type: {file_name}")
            return None

        # Convert context into LangChain retriever
        docs = [Document(page_content=context)]
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        return retriever

    except Exception as e:
        st.error(f"⚠️ Error processing {uploaded_file.name}: {str(e)}")
        return None


def get_context(query, retriever):
    """Get relevant context from retriever"""
    if retriever:
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context[:600]
        except:
            return ""
    return ""


def clean_response(response):
    """Clean model output"""
    if not response:
        return "⚠️ I couldn't generate a response. Please try again."

    response = response.strip()
    stop_tokens = ["<end_of_turn>", "</s>", "<eos>", "<|endoftext|>"]
    for token in stop_tokens:
        if token in response:
            response = response.split(token)[0].strip()

    if response and len(response) > 10:
        if not response.endswith(('.', '!', '?', ':')):
            last_punct = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_punct > len(response) // 2:
                response = response[:last_punct + 1]
            else:
                response += "."
    return response


# === UI ===
st.title("📄📊🗄️ Gemma Universal Data Chat")
st.caption("Chat with PDFs, Excel, CSV, and SQL datasets")

if not st.session_state.model_loaded:
    result = load_model()
    if result[0]:
        st.session_state.conversation, device = result
        st.session_state.model_loaded = True
        st.success(f"✅ Model loaded on {device}!")
    else:
        st.stop()

# Chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input + file upload
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.chat_input("Ask about your data or PDF...")
with col2:
    uploaded_file = st.file_uploader("📁", type=["pdf", "csv", "xls", "xlsx", "xlsm", "xlsb", "sql"], label_visibility="collapsed")

# Handle file upload
if uploaded_file and uploaded_file.name != st.session_state.file_name:
    retriever = process_file(uploaded_file)
    if retriever:
        st.session_state.retriever = retriever
        st.session_state.file_name = uploaded_file.name
        st.success(f"✅ {uploaded_file.name} ready for chat!")

# Handle chat
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                context = get_context(user_input, st.session_state.retriever)
                response = st.session_state.conversation.run(
                    human_input=user_input, context=context
                )
                response = clean_response(response)
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
