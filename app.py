import streamlit as st
import torch
import os
import pandas as pd
import io
import re
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

# Smart Excel wrapper
from smart_excel_wrapper import SmartExcelWrapper

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

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
    st.session_state.excel_wrapper = SmartExcelWrapper(model_path="./qwen_excel_gpu")

# -------------------------
# Enhanced Request Classification
# -------------------------
def classify_request_type(query):
    """More sophisticated classification for routing"""
    query_lower = query.lower()
    
    # Excel formula indicators (high precision patterns)
    formula_patterns = [
        r'formula\b', r'excel\b', r'=\w+\(', r'calculate.*total',
        r'sum of', r'count.*how many', r'percentage of', r'average of',
        r'max.*value', r'min.*value', r'total.*sales', r'count.*games'
    ]
    
    # Data analysis indicators
    analysis_patterns = [
        r'summary', r'describe', r'overview', r'structure', r'columns',
        r'what.*data', r'tell me about', r'analyze', r'insights',
        r'show me.*data', r'dataset.*contains', r'rows.*columns'
    ]
    
    # Check patterns with regex for better matching
    if any(re.search(pattern, query_lower) for pattern in formula_patterns):
        return "excel_formula"
    elif any(re.search(pattern, query_lower) for pattern in analysis_patterns):
        return "data_analysis"
    else:
        return "general"

def route_query(query, csv_loaded=False):
    """Simple router to decide which model to use"""
    request_type = classify_request_type(query)
    
    if request_type == "excel_formula" and csv_loaded:
        return "fine_tuned_model"
    else:
        return "gemma_2b"

# -------------------------
# Load LLM Model (Gemma 2B for reasoning)
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
        st.error("No Hugging Face token found!")
        return None, device

    with st.spinner("Loading Gemma 2B reasoning model..."):
        pipe = pipeline(
            "text-generation",
            model="google/gemma-2b-it",
            device_map="auto" if device.type == "cuda" else None,
            token=token,
            trust_remote_code=True,
            max_new_tokens=200,
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

        # Enhanced prompt template for better data analysis
        prompt_template = PromptTemplate(
            input_variables=["human_input", "context"],
            template="""You are a helpful data analysis assistant. You can analyze datasets, explain data patterns, and provide insights.

Dataset Context:
{context}

Instructions:
- For data analysis questions, provide clear insights based on the dataset information
- For explanations, be concise and accurate
- For general questions, provide helpful responses
- Do not generate Excel formulas (that's handled separately)

Question: {human_input}

Response:"""
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
        st.error("File too large! Please upload files under 20MB.")
        return None, None

    ext = uploaded_file.name.lower().split('.')[-1]

    if ext == "pdf":
        with st.spinner("Processing PDF..."):
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
        with st.spinner("Processing CSV..."):
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
            st.success(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            # Update Excel wrapper with CSV data
            st.session_state.excel_wrapper.update_csv(df)

            # Create documents for RAG
            docs = [
                Document(page_content=f"Dataset Structure: {df.shape[0]} rows and {df.shape[1]} columns"),
                Document(page_content=f"Column names: {', '.join(df.columns.tolist())}"),
                Document(page_content=f"Data types:\n{df.dtypes.to_string()}"),
                Document(page_content=f"Dataset preview (first 5 rows):\n{df.head().to_string()}"),
                Document(page_content=f"Dataset summary statistics:\n{df.describe().to_string()}"),
                Document(page_content=f"Missing values per column:\n{df.isnull().sum().to_string()}")
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
    """Get relevant context for the query"""
    if retriever:
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context[:800]  # Increased context length
        except:
            return ""
    return ""

# -------------------------
# Main Streamlit App
# -------------------------
def main():
    st.title("🤖 AI Data Assistant")
    st.markdown("*Powered by Gemma 2B for reasoning + Fine-tuned model for Excel formulas*")

    # Load Gemma 2B model
    if not st.session_state.model_loaded:
        result = load_model()
        if result:
            st.session_state.conversation, device = result
            st.session_state.model_loaded = True
            st.success("✅ Models loaded successfully!")

    # File status indicator
    if st.session_state.pdf_name:
        st.info(f"📄 Active file: **{st.session_state.pdf_name}**")
        if st.session_state.csv_data is not None:
            with st.expander("📊 Dataset Preview"):
                st.dataframe(st.session_state.csv_data.head(10))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(st.session_state.csv_data))
                with col2:
                    st.metric("Columns", len(st.session_state.csv_data.columns))
                with col3:
                    st.metric("Missing Values", st.session_state.csv_data.isnull().sum().sum())

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input + Upload layout
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.chat_input("Ask about your data, request Excel formulas, or general questions...")
    with col2:
        uploaded_file = st.file_uploader("📁", type=["pdf", "csv"], label_visibility="collapsed")

    # Handle file upload
    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        retriever, df = process_file(uploaded_file)
        if retriever:
            st.session_state.retriever = retriever
            st.session_state.csv_data = df
            st.session_state.pdf_name = uploaded_file.name
            st.rerun()  # Refresh to show new file status

    # Handle user input with improved routing
    if user_input and st.session_state.model_loaded:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # Route the query to appropriate model
                model_to_use = route_query(user_input, st.session_state.csv_data is not None)
                request_type = classify_request_type(user_input)
                
                if model_to_use == "fine_tuned_model" and request_type == "excel_formula":
                    # Use fine-tuned model for Excel formulas
                    excel_formula = st.session_state.excel_wrapper.generate_smart_formula(user_input)
                    
                    if excel_formula and excel_formula.startswith('=') and 'could not generate' not in excel_formula.lower():
                        # Display the formula
                        st.markdown("### 📊 Excel Formula")
                        st.code(excel_formula, language='excel')
                        
                        # Use Gemma 2B to explain the formula
                        explanation_prompt = f"""Explain this Excel formula in simple terms: {excel_formula}

Original question: {user_input}

Provide a brief, clear explanation of what this formula calculates."""
                        
                        explanation = st.session_state.conversation.run(
                            human_input=explanation_prompt,
                            context=""
                        )
                        
                        st.markdown("### 💡 Explanation")
                        st.markdown(explanation)
                        
                        full_response = f"**Excel Formula:**\n```excel\n{excel_formula}\n```\n\n**Explanation:**\n{explanation}"
                    else:
                        # Fallback to Gemma if formula generation fails
                        st.warning("Formula generation failed, using general analysis...")
                        context = get_context(user_input, st.session_state.retriever)
                        response = st.session_state.conversation.run(
                            human_input=user_input,
                            context=context
                        )
                        st.markdown(response)
                        full_response = response

                else:
                    # Use Gemma 2B for data analysis, summaries, and general questions
                    context = get_context(user_input, st.session_state.retriever)
                    response = st.session_state.conversation.run(
                        human_input=user_input,
                        context=context
                    )
                    st.markdown(response)
                    full_response = response

                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # Sidebar with helpful information
    with st.sidebar:
        st.header("🔧 Model Status")
        
        if st.session_state.model_loaded:
            st.success("✅ Gemma 2B (Reasoning)")
        else:
            st.error("❌ Gemma 2B not loaded")
            
        if st.session_state.excel_wrapper.model_loaded:
            st.success("✅ Fine-tuned (Excel)")
        else:
            st.warning("⚠️ Fine-tuned model not loaded")
        
        st.divider()
        
        st.header("💡 Usage Tips")
        st.markdown("""
        **For Excel Formulas:**
        - "Calculate the total sales"
        - "Count how many games from 2023"
        - "What percentage are Action games?"
        - "Average revenue by publisher"
        
        **For Data Analysis:**
        - "Summarize this dataset"
        - "What columns are available?"
        - "Show me data insights"
        - "Describe the data structure"
        
        **General Questions:**
        - Ask anything about the data
        - Request explanations
        - Get recommendations
        """)
        
        if st.session_state.csv_data is not None:
            st.divider()
            st.header("📊 Dataset Info")
            st.markdown(f"**Rows:** {len(st.session_state.csv_data)}")
            st.markdown(f"**Columns:** {len(st.session_state.csv_data.columns)}")
            
            with st.expander("Column Details"):
                for i, col in enumerate(st.session_state.csv_data.columns):
                    st.markdown(f"**{chr(65+i)}:** {col}")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.conversation.memory.clear()
            st.rerun()

if __name__ == "__main__":
    main()