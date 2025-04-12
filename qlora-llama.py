import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
import bitsandbytes as bnb

# Config
MODEL_NAME = "meta-llama/Llama-3.1-8b-hf"
DATA_PATH = "gait_data.jsonl"
OUTPUT_DIR = "./qlora_gait_llama"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Prompt formatting
def format_example(example):
    return {
        "text": f"### Gait Summary:\n{example['input']}\n\n### Diagnosis:\n{example['output']}"
    }

dataset = dataset.map(format_example)

# Tokenization
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Load base model (4-bit)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    quantization_config=bnb.nn.Linear4bit.get_quant_config(),
)

# Prepare for QLoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # works for LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    fp16=True,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator,
)

trainer.train()


model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
