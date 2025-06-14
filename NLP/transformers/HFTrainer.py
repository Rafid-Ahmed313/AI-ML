from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
import numpy as np, torch

# 1. toy data: 2 000 lines of AG News (finishes in <1 min)
ds = load_dataset("ag_news", split="train[:2000]")
train_ds, val_ds = ds.train_test_split(test_size=0.2, seed=42).values()

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=False)

def enc(b): return tok(b["text"], truncation=True, max_length=64)
train_ds = train_ds.map(enc, batched=True, remove_columns=["text"])
val_ds   = val_ds.map(enc,   batched=True, remove_columns=["text"])
train_ds = train_ds.rename_column("label", "labels")
val_ds   = val_ds.rename_column("label", "labels")
train_ds.set_format("torch"); val_ds.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=4)

args = TrainingArguments(
    "tmp_agnews",
    per_device_train_batch_size = 8,
    num_train_epochs            = 1,        # bump later
    logging_steps               = 10,
    evaluation_strategy         = "epoch",
    fp16                        = torch.cuda.is_available(),
    remove_unused_columns       = False,
    dataloader_num_workers      = 0,        # ←  critical on Kaggle
)

def accuracy(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

trainer = Trainer(model, args,
                  train_dataset=train_ds,
                  eval_dataset =val_ds,
                  data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=accuracy)

trainer.train()

