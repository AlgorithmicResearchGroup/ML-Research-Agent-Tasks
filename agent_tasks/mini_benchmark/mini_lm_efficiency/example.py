import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, EvalPrediction, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


dataset = load_dataset("EleutherAI/wikitext_document_level", "wikitext-103-v1", trust_remote_code=True)

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset['train'].train_test_split(train_size=0.8, test_size=0.08, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

# Load tokenizer and dataset
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')


tokenizer.pad_token = tokenizer.eos_token
# Tokenize the data
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['page'], return_tensors='pt', padding="max_length", truncation=True, max_length=256)
    # GPT-2 uses the same input_ids as labels for calculating loss
    tokenized_inputs['labels'] = tokenized_inputs.input_ids.clone()
    return tokenized_inputs

tokenized_datasets = split_dataset.map(tokenize_function, batched=True)


train_dataset = tokenized_datasets['train']
val_dataset = tokenized_datasets['val']


model = GPT2LMHeadModel.from_pretrained('AlgorithmicResearchGroup/gpt2-xs')


# Training arguments
training_args = TrainingArguments(
    output_dir='./finetuned_gpt2',          # output directory
    num_train_epochs=3,              # number of training epochs
    max_steps=5000,
    eval_steps=250,
    seed=142,
    learning_rate=5e-5,
    weight_decay=0.01,
    adam_beta1 = 0.9,
    adam_beta2 = 0.95,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    bf16=True,
    do_eval=True,
    per_device_train_batch_size=64,   # batch size for training
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    report_to=None,
    include_tokens_per_second=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()