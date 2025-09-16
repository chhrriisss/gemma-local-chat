# fine_tune_vidsales.py
import os
import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# GPU & training settings
CSV_FOLDER = "./csv_datasets"
NEW_CSV = "vidsales.csv"
MODEL_DIR = "./qwen_excel_gpu"   # previously trained model
OUTPUT_DIR = "./qwen_excel_gpu"  # update in-place
EPOCHS = 3


BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_LENGTH = 128
MAX_EXAMPLES = 1200 # max examples to generate from this CSV
SEED = 42

def main():
    # GPU check
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU detected!")
        return

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    def create_training_data():
        all_texts = []
        print(f"Creating training data from {NEW_CSV}...")
        filepath = os.path.join(CSV_FOLDER, NEW_CSV)
        try:
            df = pd.read_csv(filepath)
            df = df.head(MAX_EXAMPLES)  # limit rows for safety
            columns = list(df.columns)
            col_mapping = {col: chr(65 + i) for i, col in enumerate(columns)}
            categorical_cols = [c for c in columns if pd.api.types.is_object_dtype(df[c])]
            numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]

            csv_examples = []

            # categorical examples
            for col in categorical_cols:
                col_letter = col_mapping[col]
                unique_vals = list(df[col].dropna().unique())[:5]
                for val in unique_vals:
                    # count variations
                    for query in [f"count {val}", f"how many {val}", f"number of {val}", f"total {val}", f"{val} count"]:
                        text = f"<|im_start|>user\nDataset: {NEW_CSV}\nColumns: {columns}\nQuery: {query}<|im_end|>\n" \
                               f"<|im_start|>assistant\n=COUNTIF({col_letter}:{col_letter},\"{val}\")<|im_end|>"
                        csv_examples.append(text)
                    # percentage variations
                    for query in [f"percentage of {val}", f"percent {val}", f"what percent are {val}"]:
                        text = f"<|im_start|>user\nDataset: {NEW_CSV}\nColumns: {columns}\nQuery: {query}<|im_end|>\n" \
                               f"<|im_start|>assistant\n=COUNTIF({col_letter}:{col_letter},\"{val}\")/COUNTA({col_letter}:{col_letter})*100<|im_end|>"
                        csv_examples.append(text)

            # numeric examples
            for col in numeric_cols:
                col_letter = col_mapping[col]
                operations = {
                    "sum": ["sum", "total", "add up"],
                    "average": ["average", "mean", "avg"],
                    "max": ["maximum", "max", "highest"],
                    "min": ["minimum", "min", "lowest"]
                }
                for op, variations in operations.items():
                    excel_func = op.upper() if op != "average" else "AVERAGE"
                    for qword in variations:
                        for query in [f"{qword} {col}", f"{qword} of {col}"]:
                            text = f"<|im_start|>user\nDataset: {NEW_CSV}\nColumns: {columns}\nQuery: {query}<|im_end|>\n" \
                                   f"<|im_start|>assistant\n={excel_func}({col_letter}:{col_letter})<|im_end|>"
                            csv_examples.append(text)

            random.shuffle(csv_examples)
            all_texts.extend(csv_examples[:MAX_EXAMPLES])
            print(f"  Generated {len(csv_examples[:MAX_EXAMPLES])} examples from {NEW_CSV}")

        except Exception as e:
            print(f"Error processing {NEW_CSV}: {e}")

        random.shuffle(all_texts)
        return all_texts

    # Generate dataset
    training_texts = create_training_data()
    print(f"Created {len(training_texts)} total training examples")

    dataset = Dataset.from_dict({"text": training_texts})

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize, batched=True, batch_size=128, remove_columns=["text"])

    # Load previously trained model
    print("Loading existing model on GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
        device_map="auto",
        use_cache=False
    )
    model.gradient_checkpointing_enable()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=200,
        eval_strategy="no",
        save_strategy="steps",
        fp16=False,
        bf16=False,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
        dataloader_pin_memory=False,
        warmup_steps=50,
        max_grad_norm=1.0,
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    # Trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    torch.cuda.empty_cache()

    # Train
    print("Starting GPU training on vidsales.csv...")
    try:
        trainer.train()
        print("Training completed successfully!")
        print("Saving updated model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model updated at: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        print("GPU training completed!")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
