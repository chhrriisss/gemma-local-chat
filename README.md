emma Local Chat

A local AI chatbot using Google's Gemma-3-270M model with conversation memory.

Features



🤖 Local AI chatbot using Gemma-3-270M-IT model

💾 Conversation memory - remembers your entire chat

⚡ Fast responses optimized for CPU

🔧 Easy to customize and extend



Requirements



Python 3.8+

~2GB RAM

~1GB disk space (for model download)



Installation



Clone this repository:



bashgit clone <your-repo-url>

cd gemma-local



Create virtual environment:



bashpython -m venv .venv

.venv\\Scripts\\activate  # Windows

\# source .venv/bin/activate  # Linux/Mac



Install dependencies:



bashpip install torch transformers langchain langchain-community langchain-huggingface

Usage

Run the chatbot:

bashpython gemma.py

Commands:



Type normally to chat

clear - Reset conversation memory

exit, quit, or bye - End the chat



Model Information



Model: google/gemma-3-270m-it

Size: ~270MB

Type: Instruction-tuned conversational AI

Performance: Optimized for CPU, fast responses



Configuration

Edit gemma.py to customize:



max\_new\_tokens: Response length (default: 200)

temperature: Randomness (0.1-1.0, default: 0.7)

model\_name: Use different Gemma models



Architecture

User Input → Memory + Prompt Template → Gemma Model → Clean Response → Display

&nbsp;    ↑                                                                    ↓

&nbsp;    └─────────────── Conversation History Storage ←───────────────────────┘

Dependencies



torch - PyTorch for model inference

transformers - HuggingFace model loading

langchain - Conversation management

langchain-huggingface - Model integration



License

MIT License

Contributing

Feel free to open issues or submit pull requests!

