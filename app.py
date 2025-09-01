import streamlit as st
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Page config
st.set_page_config(
    page_title="Gemma Local Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_model():
    """Load the Gemma model - cached so it only loads once"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with st.spinner("Loading Gemma model... This may take a minute..."):
        # Create pipeline
        gemma_pipe = pipeline(
            "text-generation",
            model="google/gemma-3-270m-it",  # Your 270m model
            device_map="auto" if device.type == "cuda" else None,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            return_full_text=False,
            repetition_penalty=1.1
        )
        
        # Create LangChain pipeline
        gemma_llm = HuggingFacePipeline(pipeline=gemma_pipe)
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            input_key="human_input"
        )
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["chat_history", "human_input"],
            template="""<bos>Previous conversation:
{chat_history}

<start_of_turn>user
{human_input}<end_of_turn>
<start_of_turn>model
"""
        )
        
        # Create conversation chain
        conversation = LLMChain(
            llm=gemma_llm,
            prompt=prompt_template,
            memory=memory,
            verbose=False
        )
        
        return conversation, device

def clean_response(response):
    """Clean up the model response"""
    response = response.strip()
    if response.endswith("<end_of_turn>"):
        response = response[:-13].strip()
    if response.endswith("</s>"):
        response = response[:-4].strip()
    return response

# Title and description
st.title("Gemma Local Chat")

# Load model
if not st.session_state.model_loaded:
    try:
        st.session_state.conversation, device = load_model()
        st.session_state.model_loaded = True
        st.success(f"✅ Model loaded successfully on {device}!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Chat interface

# Display chat history
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask Gemma anything...")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation.run(human_input=user_input)
                response = clean_response(response)
                
                # Display response
                st.write(response)
                
                # Add to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
