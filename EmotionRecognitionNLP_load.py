import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("sentiment_model")
model = DistilBertForSequenceClassification.from_pretrained("sentiment_model").to(device)

# prediction function
def predict_batch(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    for txt, label in zip(text_list, preds):
        print(f"text: {txt} => emotion: {'positive' if label == 1 else 'negative'}")

# test
sample_texts = [
    "I just got promoted in my job!",
    "This meal is disgusting.",
    "I'm worried about my future."
]
predict_batch(sample_texts)
