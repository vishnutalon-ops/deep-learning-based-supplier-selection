import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# Load dataset
df = pd.read_csv(r"datasetlocation")

# Keep only required columns
df = df[["Text", "Score"]].dropna()

# Convert to binary sentiment
df = df[df["Score"] != 3]  # remove neutral
df["label"] = df["Score"].apply(lambda x: 1 if x >= 4 else 0)

# Sample 10,000 reviews
df = df.sample(n=10000, random_state=42)

# Train / validation split
train_df, val_df = train_test_split(
    df[["Text", "label"]],
    test_size=0.2,
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(
        batch["Text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./roberta-finetuned",
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",      # ✅ ADD THIS
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to=[]
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./roberta-finetuned")
tokenizer.save_pretrained("./roberta-finetuned")

