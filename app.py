import streamlit as st
import torch
import os
import pandas as pd
import io
from transformers import pipeline

# LangChain imports (compatible with older versions)
try:
    # Try new structure first
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    # Fallback to old structure
    from langchain.llms import HuggingFacePipeline
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.document_loaders import PyPDFLoader

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import UI styling
from ui_styles import apply_custom_styling, render_header, render_status_card

# Smart Excel wrapper (your updated comprehensive version)
from smart_excel_wrapper import SmartExcelWrapper

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="AI Data Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_styling()

# -------------------------
# Session State
# -------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "excel_wrapper" not in st.session_state:
    st.session_state.excel_wrapper = SmartExcelWrapper(model_path="./qwen_excel_reasoning_gpu")

# -------------------------
# Helper functions for query detection
# -------------------------
def is_formula_request(query):
    """Detect if query is asking for an Excel formula"""
    formula_keywords = [
        'formula', 'excel', 'calculate', 'count', 'sum', 'average', 'percentage',
        'percent', 'total', 'max', 'min', 'how many', 'number of', 'appears',
        'occurs', 'times does', 'frequency'
    ]
    return any(keyword in query.lower() for keyword in formula_keywords)

def is_summary_request(query):
    """Detect if query is asking for dataset summary"""
    summary_keywords = ['summary', 'summarize', 'describe', 'overview', 'structure', 'columns', 'what data', "what's in"]
    return any(keyword in query.lower() for keyword in summary_keywords)

def is_csv_query(query):
    """Detect if query is about CSV data analysis"""
    return is_formula_request(query) or is_summary_request(query)

# -------------------------
# Load LLM Model (Silent Loading)
# -------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_path = os.path.expanduser('~/.cache/huggingface/token')
    token = None
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = f.read().strip()
    else:
        st.error("🚫 No Hugging Face token found!")
        return None, device

    with st.spinner("🧠 Loading AI model..."):
        pipe = pipeline(
            "text-generation",
            model="google/gemma-2b-it",
            device_map="auto" if device.type == "cuda" else None,
            token=token,
            trust_remote_code=True,
            max_new_tokens=150,
            min_new_tokens=10,
            temperature=0.3,
            do_sample=False,
            return_full_text=False
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="human_input"
        )

        prompt_template = PromptTemplate(
            input_variables=["human_input", "context"],
            template="""
You are an expert data assistant.
You have access to the dataset provided below:

Dataset:
{context}

Instructions:
- For general questions about the data, provide helpful insights.
- Be concise and accurate.
- If asked about data structure, refer to the columns and data types shown.
- Answer questions about the data content, patterns, and insights.
- For Excel formulas and calculations, defer to the specialized Excel system.

Question/Request: {human_input}

Answer:
"""
        )

        conversation = LLMChain(
            llm=llm,
            prompt=prompt_template,
            memory=memory,
            verbose=False
        )

        return conversation, device

# -------------------------
# Process CSV or PDF
# -------------------------
def process_file(uploaded_file):
    if uploaded_file.size > 20 * 1024 * 1024:
        st.error("🚫 File too large! Please use files under 20MB.")
        return None, None

    ext = uploaded_file.name.lower().split('.')[-1]

    if ext == "pdf":
        with st.spinner("📄 Processing PDF..."):
            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            docs = PyPDFLoader(temp_path).load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return retriever, None

    elif ext == "csv":
        with st.spinner("📊 Processing CSV..."):
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
            st.success(f"✅ CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            st.session_state.excel_wrapper.update_csv(df)

            docs = [
                Document(page_content=f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns."),
                Document(page_content=f"Column names: {', '.join(df.columns.tolist())}"),
                Document(page_content=f"Data types: {df.dtypes.to_string()}"),
                Document(page_content=f"First 5 rows:\n{df.head().to_string()}"),
                Document(page_content=f"Last 5 rows:\n{df.tail().to_string()}")
            ]
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            return retriever, df

    return None, None

# -------------------------
# Get Context for LLM
# -------------------------
def get_context(query, retriever):
    if retriever:
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context[:600]
        except:
            return ""
    return ""

# -------------------------
# Main Streamlit App (Beautiful UI)
# -------------------------
def main():
    # Render beautiful header
    render_header()

    # Modern sidebar
    with st.sidebar:
        st.markdown("# 🎛️ Control Panel")
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Restart", help="Clear everything and restart", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("💬 Clear Chat", help="Clear chat history only", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        # Status panel
        st.markdown("### 📊 System Status")
        
        # Model status
        if st.session_state.get("model_loaded", False):
            render_status_card("success", "✅ AI Model Ready")
        else:
            render_status_card("warning", "⏳ Loading AI Model...")
            
        # CSV status
        if st.session_state.get("csv_data") is not None:
            rows = st.session_state.csv_data.shape[0]
            cols = st.session_state.csv_data.shape[1]
            render_status_card("success", f"📊 CSV: {rows:,} rows × {cols} cols")
        else:
            render_status_card("info", "📄 No CSV loaded")
        
        # Dataset metrics
        if st.session_state.get("csv_data") is not None:
            st.markdown("---")
            st.markdown("### 📈 Dataset Metrics")
            
            df = st.session_state.csv_data
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", df.shape[1])
            
            # Column types
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            text_cols = len(df.select_dtypes(include=['object']).columns)
            
            if numeric_cols > 0:
                st.metric("Numeric", numeric_cols)
            if text_cols > 0:
                st.metric("Text", text_cols)

    # Load LLM 
    if not st.session_state.model_loaded:
        result = load_model()
        if result:
            st.session_state.conversation, device = result
            st.session_state.model_loaded = True

    # File upload section
    st.markdown("### 📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "csv"],
        help="Upload CSV for data analysis or PDF for document chat"
    )

    # Dataset preview
    if st.session_state.get("csv_data") is not None:
        with st.expander("📊 Dataset Preview", expanded=False):
            tab1, tab2 = st.tabs(["📋 Data", "📈 Info"])
            
            with tab1:
                st.dataframe(st.session_state.csv_data.head(10), use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                df = st.session_state.csv_data
                
                with col1:
                    st.markdown("**Column Names:**")
                    for i, col in enumerate(df.columns, 1):
                        st.write(f"{i}. {col}")
                
                with col2:
                    st.markdown("**Data Types:**")
                    for col, dtype in df.dtypes.items():
                        st.write(f"• {col}: {dtype}")

    # Chat section
    st.markdown("### 💬 Chat")
    
    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about your data...")

    # Handle file upload 
    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        retriever, df = process_file(uploaded_file)
        if retriever:
            st.session_state.retriever = retriever
            st.session_state.csv_data = df
            st.session_state.pdf_name = uploaded_file.name

    # Handle user input
    if user_input and st.session_state.model_loaded:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Check if CSV is loaded and query is about CSV data
                if st.session_state.csv_data is not None and is_csv_query(user_input):
                    
                    # For summary requests, use Gemma directly (it's better at summaries)
                    if is_summary_request(user_input):
                        st.markdown("### 📊 Dataset Summary")
                        context = get_context(user_input, st.session_state.retriever)
                        response = st.session_state.conversation.run(
                            human_input=f"Provide a detailed summary of this dataset including column names, data types, number of rows, and key insights from the data shown.",
                            context=context
                        )
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # For formula requests, use the fine-tuned model with fallback
                    elif is_formula_request(user_input):
                        response = st.session_state.excel_wrapper.generate_comprehensive_answer(user_input)
                        
                        # Check if response is generic/bad (has placeholder text)
                        if ("[rows]" in response or "[columns]" in response or 
                            "This dataset contains" in response and ("rows] rows" in response or "columns] columns" in response)):
                            # Fall back to Gemma for better response
                            context = get_context(user_input, st.session_state.retriever)
                            fallback_response = st.session_state.conversation.run(
                                human_input=f"Help create an Excel formula for this request: {user_input}. Provide the formula and explain how it works.",
                                context=context
                            )
                            st.markdown(fallback_response)
                            st.session_state.chat_history.append({"role": "assistant", "content": fallback_response})
                        else:
                            # Display the fine-tuned model response
                            st.markdown(response)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})

                else:
                    # Use regular LLM for non-CSV queries or when no CSV is loaded
                    context = get_context(user_input, st.session_state.retriever)
                    response = st.session_state.conversation.run(
                        human_input=user_input,
                        context=context[:400] if context else ""
                    )
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()