from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model and tokenizer loaded successfully!")
