# notebooks/fine_tune_transformer.py

import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 0. Define Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# 1. Load Data
print("Loading data...")
data = pd.read_csv('../train.csv')  # Adjust if needed based on your structure

# 2. Select relevant columns
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
data = data[['comment_text'] + label_cols]

# 3. Create small sample for fast training
print("Sampling 200 rows for quick fine-tuning...")
data = data.sample(n=200, random_state=42).reset_index(drop=True)

# 4. Preprocess labels for multi-label classification
data['labels'] = data[label_cols].values.tolist()

# 5. Train/Test Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['comment_text'].tolist(), data['labels'].tolist(), test_size=0.2, random_state=42
)


# 6. Tokenization
print("Tokenizing data...")
model_checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# 7. Define Model
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)

# 8. Define Training Arguments
training_args = TrainingArguments(
    output_dir="../models/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="../models/logs",
    logging_steps=10,
    save_total_limit=2,
)

# 9. Define Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(logits)) > 0.5
    return {
        "macro_f1": classification_report(labels, predictions.numpy(), output_dict=True)['macro avg']['f1-score']
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 10. Train
print("Starting training...")
trainer.train()

# # 11. Save Model
# model_save_path = "../models/saved_model_roberta_toxic"
# trainer.save_model(model_save_path)
# # Save model and tokenizer in HuggingFace format
# model.save_pretrained("../models/saved_model_roberta_toxic")
# tokenizer.save_pretrained("../models/saved_model_roberta_toxic")

import os

# Correct safe path
model_save_path = os.path.join(os.getcwd(), "models", "saved_model_roberta_toxic")
os.makedirs(model_save_path, exist_ok=True)

trainer.save_model(model_save_path)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)



print(f"\n✅ Model and tokenizer saved to {model_save_path}")
