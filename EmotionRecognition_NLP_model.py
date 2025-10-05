import torch
import pandas as pd
from datasets import Dataset
from datasets import ClassLabel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import re

#GPU setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use of : {device}")

# read CSV and change it to UTF-8
print("📂 resd CSV by encoding ISO-8859-1...")
df = pd.read_csv(
    'training.1600000.processed.noemoticon.csv',
    encoding='ISO-8859-1',
    names=["target","id","date","flag","user","text"]
)

print(f"✅ records: {len(df)}")

# Choose a small sample for rapid testing
df = df.sample(n=200000, random_state=42).reset_index(drop=True)
print(f"📊 selected sample {len(df)} record")

# text preprocessing
def clean_text(text):
    text = re.sub(r"http\S+", "", text)       # remove link
    text = re.sub(r"@\w+", "", text)          # remove username
    text = re.sub(r"[^A-Za-z\s]", "", text)   # remove non-letter characters 
    return text.strip()

df['text'] = df['text'].apply(clean_text)

# convert to class labels
class_labels = ClassLabel(names=["negative","positive"])
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("target", class_labels)
dataset = dataset.rename_column("target", "labels")

#Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["id","date","flag","user","text"])
dataset.set_format("torch")

# data split
train_test = dataset.train_test_split(test_size=0.2)

# load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

# metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1}

# train setting
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",           
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=200,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test['train'],
    eval_dataset=train_test['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#  train
print("🚀 start training ...")
trainer.train()

# save model
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")
print("✅ tokenizer and model saved.")

# test
def predict_batch(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    for txt, label in zip(text_list, preds):
        print(f"text: {txt} => emotion : {'positive' if label==1 else 'negative'}")

sample_texts = [
    "I don't love these cakes",
    "I said yes and I enggaged.",
    "such a poor man..."
]
predict_batch(sample_texts)

# evaluate
final_eval = trainer.evaluate()
print(f"📈 final evaluate results : {final_eval}")
