import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

# Load the babylm dataset
dataset = load_dataset("AlgorithmicResearchGroup/babylm")

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the data
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['content'], return_tensors='pt', padding="max_length", truncation=True, max_length=256)
    tokenized_inputs['labels'] = tokenized_inputs.input_ids.clone()
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['filename', 'content'])

# Custom GPT-2 configuration: 6-layer model (very small model)
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2
)
model = GPT2LMHeadModel(config)

# Training arguments
training_args = TrainingArguments(
    output_dir='./gpt2_babylm_model',
    max_steps=5000,
    eval_steps=250,
    seed=142,
    learning_rate=1e-3,
    weight_decay=1e-1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    bf16=True,
    do_eval=True,
    per_device_train_batch_size=64,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="steps",
    save_steps=250,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    report_to=None,
    include_tokens_per_second=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['dev'],
    tokenizer=tokenizer
)

# Start training
trainer.train()