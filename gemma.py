import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Use the smaller, faster model (your original choice)
model_name = "google/gemma-3-270m-it"
print(f"Loading {model_name}...")

# Create pipeline directly (simpler and faster)
gemma_pipe = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto" if device.type == "cuda" else None,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    return_full_text=False,
    repetition_penalty=1.1
)

print("Model loaded successfully!")

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

def clean_response(response):
    """Clean up the model response"""
    response = response.strip()
    if response.endswith("<end_of_turn>"):
        response = response[:-13].strip()
    if response.endswith("</s>"):
        response = response[:-4].strip()
    return response

# Chat loop
print("Gemma Chat with Memory — type 'exit' to quit")
print("Using lightweight 270m model for faster responses!")
print("-" * 50)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
    if user_input.lower() == "clear":
        memory.clear()
        print("Memory cleared!")
        continue
    
    try:
        response = conversation.run(human_input=user_input)
        response = clean_response(response)
        print(f"Gemma: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again...")

print("Chat ended.")