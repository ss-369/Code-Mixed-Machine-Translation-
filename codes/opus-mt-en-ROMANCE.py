# Install necessary packages
!pip install transformers datasets sacrebleu sentencepiece evaluate

# Import necessary libraries
import os
import torch
import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import random

# Set environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check if GPUs are available and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {n_gpu}")
else:
    device = torch.device("cpu")
    n_gpu = 0
    print("No GPU available, using CPU.")

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

# Timing: Start
start_time = time.time()

# Initialize model, tokenizer, and config
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)

# Move model to device before wrapping with DataParallel
model.to(device)

# Wrap the model with DataParallel if multiple GPUs are available
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    print("Using DataParallel for multi-GPU training.")

# Load the dataset
dataset_load_start = time.time()
dataset = load_dataset('findnitai/english-to-hinglish')
dataset = dataset['train']
dataset = datasets.DatasetDict({'train': dataset})
raw_datasets = dataset['train'].train_test_split(test_size=0.1, seed=seed)
train_dataset = raw_datasets['train']
eval_dataset = raw_datasets['test']
print(f"Dataset loaded in {time.time() - dataset_load_start:.2f} seconds")

# Print dataset sizes
print(f"Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}")

# Preprocessing parameters
source_prefix = "translate English to Hinglish: "
source_lang = "en"
target_lang = "hi_ng"
max_source_length = 128
max_target_length = 128
padding = "max_length"
num_epochs = 3
gradient_accumulation_steps = 2

# Preprocessing function
def preprocess_function(examples):
    inputs = [ex[source_lang].lower().strip() for ex in examples['translation']]
    targets = [ex[target_lang].lower().strip() for ex in examples['translation']]
    inputs = [source_prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to datasets
preprocess_start = time.time()
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
print(f"Data preprocessed in {time.time() - preprocess_start:.2f} seconds")

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Create DataLoaders with adjusted batch size
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=32,  # Adjust batch size based on memory
)
eval_dataloader = DataLoader(
    eval_dataset,
    collate_fn=data_collator,
    batch_size=32,  # Larger batch size for evaluation
)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=3e-5)
total_steps = len(train_dataloader) // gradient_accumulation_steps * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# Initialize GradScaler for mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Metric for BLEU score
metric = load_metric('sacrebleu')

# Function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

# Training loop with gradient accumulation and mixed precision
total_train_loss = []
total_eval_bleu = []

print("Starting training...")
for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    epoch_train_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean() / gradient_accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        loss_value = loss.item() * gradient_accumulation_steps
        epoch_train_loss += loss_value

        if step % 1000 == 0:
            current_loss = epoch_train_loss / (step + 1)
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(train_dataloader)}], Loss: {current_loss:.4f}")

    avg_train_loss = epoch_train_loss / len(train_dataloader)
    total_train_loss.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] training completed in {time.time() - epoch_start:.2f} seconds")

    # Evaluation
    eval_start = time.time()
    model.eval()
    all_preds = []
    all_labels = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            with autocast():
                generated_tokens = model.module.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=max_target_length,
                    num_beams=4,
                )
        labels = batch["labels"]

        labels = labels.cpu().numpy()
        generated_tokens = generated_tokens.cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)

    result = metric.compute(predictions=all_preds, references=all_labels)
    eval_bleu = result["score"]
    print(f"Epoch [{epoch+1}/{num_epochs}], Evaluation BLEU Score: {eval_bleu:.2f}")
    print(f"Evaluation completed in {time.time() - eval_start:.2f} seconds")
    total_eval_bleu.append(eval_bleu)

print("Training completed.")
print(f"Total training time: {time.time() - start_time:.2f} seconds")

# Plot training loss and evaluation BLEU score
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), total_train_loss, label="Training Loss")
plt.plot(range(1, num_epochs + 1), total_eval_bleu, label="Evaluation BLEU Score")
plt.xlabel("Epoch")
plt.ylabel("Loss / BLEU Score")
plt.title("Training Loss and Evaluation BLEU Score")
plt.legend()
plt.show()

# Generate predictions on the test set
print("Generating predictions on the test set...")
model.eval()
all_preds = []
all_labels = []
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        with autocast():
            generated_tokens = model.module.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_target_length,
                num_beams=4,
            )
    labels = batch["labels"]

    labels = labels.cpu().numpy()
    generated_tokens = generated_tokens.cpu().numpy()
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    all_preds.extend(decoded_preds)
    all_labels.extend(decoded_labels)

# Compute individual BLEU scores
print("Computing BLEU scores for each test sample...")
individual_bleu_scores = [metric.compute(predictions=[pred], references=[[label]])["score"]
                          for pred, label in zip(all_preds, all_labels)]

# Plot BLEU score distribution
plt.figure(figsize=(10, 6))
sns.histplot(individual_bleu_scores, bins=20, kde=True)
plt.title("BLEU Score Distribution on Test Set")
plt.xlabel("BLEU Score")
plt.ylabel("Number of Samples")
plt.show()

# Print average BLEU score
average_bleu = np.mean(individual_bleu_scores)
print(f"Average BLEU score on test set: {average_bleu:.2f}")

# Sample translations
print("\nSample translations from the test set:")
for i in range(5):
    input_ids = eval_dataset[i]["input_ids"]
    input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(f"\nSource: {input_text.replace(source_prefix, '')}")
    print(f"Reference: {all_labels[i][0]}")
    print(f"Prediction: {all_preds[i]}")

# Function to translate new sentences
def translate_sentence(sentence):
    input_text = source_prefix + sentence.lower().strip()
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        with autocast():
            generated_tokens = model.module.generate(
                input_ids,
                max_length=max_target_length,
                num_beams=4,
            )
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text

# Test the model with a custom sentence
custom_sentence = "I was waiting for my bag"
translated_sentence = translate_sentence(custom_sentence)
print("\nCustom Translation:")
print(f"Input: {custom_sentence}")
print(f"Translated Output: {translated_sentence}")

# Optional: Monitor GPU Memory Usage
print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
