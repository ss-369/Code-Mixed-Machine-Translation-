# Install necessary packages
!pip install transformers datasets sacrebleu sentencepiece evaluate peft matplotlib seaborn

# Import necessary libraries
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datasets import load_dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    AdamW,
    get_scheduler
)
from peft import get_peft_model, LoraConfig, TaskType
from evaluate import load as load_metric
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Set environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to count trainable parameters
def count_trainable_parameters(model):
    """
    Counts the number of trainable parameters in the model.
    If the model is wrapped with DataParallel, it accesses the underlying module.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define helper functions to save model and tokenizer
def save_model(model, save_path):
    """
    Saves the model's state_dict to the specified path.
    
    Args:
        model (torch.nn.Module): The trained model.
        save_path (str): The file path to save the model.
    """
    # If model is wrapped in DataParallel, unwrap it
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
    
    # Save the state_dict
    torch.save(model_to_save.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def save_tokenizer(tokenizer, save_dir):
    """
    Saves the tokenizer to the specified directory.
    
    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to save.
        save_dir (str): The directory path to save the tokenizer.
    """
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to {save_dir}")

# Helper function to call generate correctly
def generate(model, **kwargs):
    """
    Wrapper function to call generate on the underlying model.
    Supports both DataParallel and single GPU/CPU setups.
    
    Args:
        model (torch.nn.Module): The model, potentially wrapped with DataParallel.
        **kwargs: Keyword arguments to pass to the generate method.
    
    Returns:
        torch.Tensor: Generated sequences.
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module.generate(**kwargs)
    else:
        return model.generate(**kwargs)

# Check if GPUs are available and set device
start_time = time.time()
if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {n_gpu}")
else:
    device = torch.device("cpu")
    n_gpu = 0
    print("No GPU available, using CPU.")
print(f"Device setup time: {time.time() - start_time:.2f} seconds\n")

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

# Load the dataset
start_time = time.time()
dataset = load_dataset('findnitai/english-to-hinglish')
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds\n")

# Reduce the dataset size to 50%
sample_fraction = 0.5
start_time = time.time()
dataset = dataset['train'].shuffle(seed=seed).select(range(int(len(dataset['train']) * sample_fraction)))
print(f"Dataset reduced to {sample_fraction*100}% of original size in {time.time() - start_time:.2f} seconds\n")

# Split the dataset into train, validation, and test sets
start_time = time.time()
train_test_split = dataset.train_test_split(test_size=0.2, seed=seed)
test_valid_split = train_test_split['test'].train_test_split(test_size=0.5, seed=seed)
train_dataset = train_test_split['train']
eval_dataset = test_valid_split['test']
test_dataset = test_valid_split['train']
print(f"Dataset split into train, validation, and test sets in {time.time() - start_time:.2f} seconds")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(eval_dataset)}")
print(f"Test set size: {len(test_dataset)}\n")

# Initialize tokenizer and model for English-to-Hinglish translation
start_time = time.time()
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="hi_IN")
model = MBartForConditionalGeneration.from_pretrained(model_name)
print(f"Tokenizer and model loaded in {time.time() - start_time:.2f} seconds\n")

# Define LoRA config with expanded target modules for translation improvement
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  # Rank of the update matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout probability
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
)
model = get_peft_model(model, lora_config)

# Move model to device before wrapping with DataParallel
start_time = time.time()
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    print("Using DataParallel for multi-GPU training.")
print(f"Model moved to device and wrapped with DataParallel in {time.time() - start_time:.2f} seconds\n")

# **Print the number of trainable parameters**
num_trainable_params = count_trainable_parameters(model)
print(f"Number of trainable parameters: {num_trainable_params:,}\n")

# Preprocessing function for English-to-Hinglish with max_length set to 24
max_length = 24  # Reduced for memory efficiency

def preprocess_function(examples):
    inputs = [ex['en'] for ex in examples['translation']]  # English as input
    targets = [ex['hi_ng'] for ex in examples['translation']]  # Hinglish as target
    inputs = ["translate English to Hinglish: " + ex for ex in inputs]  # Adding task prefix
    
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to datasets
start_time = time.time()
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
print(f"Datasets preprocessed in {time.time() - start_time:.2f} seconds\n")

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Dataloaders with reduced batch size and pin_memory
start_time = time.time()
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=16, pin_memory=True)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=16, pin_memory=True)
test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=16, pin_memory=True)
print(f"Dataloaders created in {time.time() - start_time:.2f} seconds\n")

# Optimizer and Scheduler
start_time = time.time()
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3  # 3 epochs
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
print(f"Optimizer and scheduler set up in {time.time() - start_time:.2f} seconds\n")

# Metric
metric = load_metric('sacrebleu')

# Variables to store training and validation results
training_losses = []
validation_bleu_scores = []
scaler = GradScaler()  # For mixed-precision training

# Custom training loop with gradient accumulation and mixed-precision
gradient_accumulation_steps = 2  # Accumulate gradients over 2 steps
num_epochs = 3
print("Starting training...\n")
total_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        step_start_time = time.time()
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with autocast():  # Enable mixed precision
            outputs = model(**batch)
            loss = outputs.loss.mean() / gradient_accumulation_steps  # Scale loss for accumulation
            epoch_loss += loss.item() * gradient_accumulation_steps

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Perform optimizer step and reset gradients after accumulation steps
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
        
        # Logging every 500 steps
        if step % 500 == 0:
            avg_loss = epoch_loss / (step + 1)
            current_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(train_dataloader)}], Loss: {avg_loss:.4f}, Time Elapsed: {current_time:.2f}s")
        
        # Logging step completion time
        if step % 500 == 0: 
            print(f"Step {step+1}/{len(train_dataloader)} completed in {time.time() - step_start_time:.2f} seconds")
    
    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    training_losses.append(avg_epoch_loss)
    print(f"\nEpoch {epoch+1} completed in {time.time() - epoch_start_time:.2f} seconds")
    print(f"Average Training Loss: {avg_epoch_loss:.4f}")
    
    # Validation at the end of each epoch
    val_start_time = time.time()
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Use the helper generate function
            outputs = generate(
                model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_length,
                num_beams=2
            )
            labels = batch["labels"]
            labels = np.where(labels.cpu().numpy() != -100, labels.cpu().numpy(), tokenizer.pad_token_id)
            all_preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
            all_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
    result = metric.compute(predictions=all_preds, references=[[label] for label in all_labels])
    validation_bleu_scores.append(result['score'])
    print(f"Validation BLEU Score after Epoch {epoch+1}: {result['score']:.2f}")
    print(f"Validation completed in {time.time() - val_start_time:.2f} seconds\n")

print(f"Total training time: {time.time() - total_start_time:.2f} seconds")
print("Training completed.\n")

# Plot training loss and validation BLEU score
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), training_losses, marker='o', label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), validation_bleu_scores, marker='o', color='orange', label="Validation BLEU Score")
plt.xlabel("Epoch")
plt.ylabel("BLEU Score")
plt.title("Validation BLEU Score over Epochs")
plt.legend()
plt.tight_layout()
plt.show()

# Test Evaluation and BLEU Score Distribution
print("Evaluating on the test set...\n")
test_start_time = time.time()
model.eval()
all_preds = []
all_labels = []
individual_bleu_scores = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        # Use the helper generate function
        outputs = generate(
            model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=max_length,
            num_beams=2
        )
        labels = batch["labels"]
        labels = np.where(labels.cpu().numpy() != -100, labels.cpu().numpy(), tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        all_preds.extend(decoded_preds)
        all_labels.extend(decoded_labels)
        
        # Calculate individual BLEU scores for each prediction
        for pred, label in zip(decoded_preds, decoded_labels):
            bleu = metric.compute(predictions=[pred], references=[[label]])
            individual_bleu_scores.append(bleu["score"])

test_result = metric.compute(predictions=all_preds, references=[[label] for label in all_labels])
print(f"Test BLEU Score: {test_result['score']:.2f}")
print(f"Test evaluation completed in {time.time() - test_start_time:.2f} seconds\n")

# Plot BLEU score distribution with density curve and mean line
plt.figure(figsize=(10, 6))
sns.histplot(individual_bleu_scores, bins=20, kde=True, color='skyblue', edgecolor='black', alpha=0.7)
mean_bleu = np.mean(individual_bleu_scores)
plt.axvline(mean_bleu, color='red', linestyle='--', label=f'Mean BLEU Score: {mean_bleu:.2f}')
plt.title("BLEU Score Distribution on Test Set")
plt.xlabel("BLEU Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()



# **Save the trained model and tokenizer**
# Define save paths
model_save_path = "english_to_hinglish_model.pt"  # You can change this path as needed
tokenizer_save_dir = "tokenizer/"  # Directory to save the tokenizer

# Call the save functions
save_model(model, model_save_path)
save_tokenizer(tokenizer, tokenizer_save_dir)



# Function to display sample translations from the test set
def display_sample_translations(num_samples=5):
    sampled_indices = random.sample(range(len(test_dataset)), num_samples)
    
    for idx in sampled_indices:
        try:
            input_text = test_dataset[idx]['en']
            reference_text = test_dataset[idx]['hi_ng']
        except KeyError as e:
            print(f"KeyError: {e}. Available keys: {list(test_dataset[idx].keys())}")
            continue
        
        input_ids = tokenizer(
            "translate English to Hinglish: " + input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).input_ids.to(device)
        
        with torch.no_grad():
            # Use the helper generate function
            generated_tokens = generate(
                model,
                input_ids=input_ids,
                max_length=max_length,
                num_beams=2
            )
        predicted_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        print("\nSample Translation:")
        print(f"Input (English): {input_text}")
        print(f"Reference (Hinglish): {reference_text}")
        print(f"Prediction (Model Output): {predicted_text}")

# Display 5 sample translations from the test set
display_sample_translations(num_samples=5)

# Function to translate a custom input sentence
def translate_sentence(sentence):
    input_text = "translate English to Hinglish: " + sentence
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).input_ids.to(device)
    with torch.no_grad():
        # Use the helper generate function
        generated_tokens = generate(
            model,
            input_ids=input_ids,
            max_length=max_length,
            num_beams=2
        )
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text

# Test the model with a custom sentence
custom_sentence = "I was waiting for my bag"
translated_sentence = translate_sentence(custom_sentence)
print("\nCustom Translation:")
print(f"Input: {custom_sentence}")
print(f"Translated Output: {translated_sentence}")
