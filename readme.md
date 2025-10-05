# Emotion Recognition using DistilBERT on Sentiment140

## 📌 Overview
This project fine-tunes **DistilBERT** to classify tweets from the **Sentiment140** dataset into two sentiments:  
`positive` or `negative`.

---

## 🎯 Objectives
- Train a robust sentiment classifier on noisy Twitter data.
- Achieve high accuracy & F1-score (~83%).
- Provide separate scripts for training and inference.
- Save and load the model without retraining.

---

## 📊 Dataset
- **Name:** Sentiment140  
- **Original size:** 1.6M tweets  
- **Sample used:** 200,000 random tweets for fast training  
- **Encoding:** Loaded with `ISO-8859-1`  
- **Labels:**  
  - `0` → negative  
  - `4` → positive → converted to `[0, 1]`

---

## ⚙️ Preprocessing Steps
1. Remove URLs  
2. Remove usernames (`@username`)  
3. Remove non-alphabetic characters (emojis, punctuation)  
4. Trim spaces

---

## 🛠️ Model & Training
- **Base model:** `distilbert-base-uncased`  
- **Epochs:** 3  
- **Batch size:** 16  
- **Optimizer:** AdamW (`weight_decay=0.01`)  
- **Evaluation strategy:** epoch  
- **Metrics:** Accuracy & F1-score  

---

## 📈 Final Results
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8306  |
| F1-score   | 0.8293  |
| Eval Loss  | 0.3824  |

---

## 🚀 Usage Guide

### 1️⃣ Training the model  
Run:
```bash
python EmotionRecognition_NLP_model.py
```
#This will:

- Load & preprocess the dataset
- Train the DistilBERT model
- Save the final model and tokenizer in the folder sentiment_model/


#Example training output:

text: I don't love these cakes => emotion: negative
text: I said yes and I enggaged. => emotion: positive
text: such a poor man... => emotion: negative

### 2️⃣ Inference with saved model (No retraining required)
The script EmotionRecognitionNLP_load.py:

Automatically loads the pre-trained DistilBERT model from sentiment_model/.
Runs sentiment prediction directly.
No training is required for inference.


Run:
```bash
python EmotionRecognitionNLP_load.py
```

Example output:

text: I just got promoted in my job! => emotion: positive
text: This meal is disgusting. => emotion: negative
text: I say no => emotion: positive


### 📂 Project Structure
dataset/
├── sentiment_model/                     # Saved model & tokenizer
├── EmotionRecognition_NLP_model.py      # Training script
├── EmotionRecognitionNLP_load.py        # Inference script
├── training.1600000.processed.noemoticon.csv  # Dataset (Sentiment140)
├── logs/                                 # Training logs
├── results/                              # Checkpoints
└── readme.md                             # Documentation

### 🔧 Installation
Install requirements before running scripts:
pip install torch transformers datasets scikit-learn tqdm

