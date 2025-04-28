# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import math
import os
import pickle
import torch.nn as nn

app = FastAPI()

# Define model paths
roberta_path = os.path.join(os.getcwd(), "models", "saved_model_roberta_toxic")
bart_path = os.path.join(os.getcwd(), "models", "saved_model_bart_toxic")
cnn_path = os.path.join(os.getcwd(), "models", "saved_model_cnn")

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 🔵 Load RoBERTa
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_path, local_files_only=True)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path, local_files_only=True)

# 🔵 Load BART
bart_tokenizer = AutoTokenizer.from_pretrained(bart_path, local_files_only=True)
bart_model = AutoModelForSequenceClassification.from_pretrained(bart_path, local_files_only=True)

# 🔵 CNN model (optional)
class CNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 100, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(100, output_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out

# Load CNN model if exists
if os.path.exists(os.path.join(cnn_path, "cnn_model.pt")):
    cnn_model = CNNClassifier(input_dim=5000, output_dim=6)
    cnn_model.load_state_dict(torch.load(os.path.join(cnn_path, "cnn_model.pt"), map_location=torch.device('cpu')))
    cnn_model.eval()
    with open(os.path.join(cnn_path, "tfidf_vectorizer.pkl"), "rb") as f:
        cnn_vectorizer = pickle.load(f)
else:
    cnn_model = None
    cnn_vectorizer = None

# 📝 Request schemas
class TextInput(BaseModel):
    text: str
    model_name: str

class BatchInput(BaseModel):
    texts: List[str]
    model_name: str

# 🧠 Prediction helper
def predict_single(text: str, model_name: str):
    model_name = model_name.lower()

    if model_name == "roberta":
        tokenizer = roberta_tokenizer
        model = roberta_model
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().tolist()

    elif model_name == "bart":
        tokenizer = bart_tokenizer
        model = bart_model
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().tolist()

    elif model_name == "cnn" and cnn_model is not None:
        input_vec = cnn_vectorizer.transform([text]).toarray()
        input_tensor = torch.tensor(input_vec, dtype=torch.float32)

        with torch.no_grad():
            outputs = cnn_model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().tolist()

    else:
        return None

    prediction = {}
    for label, prob in zip(labels, probs):
        if not math.isnan(prob) and not math.isinf(prob):
            if prob >= 0.5:
                prediction[label] = float(prob)
    return prediction

# 🚀 Single text predict
@app.post("/predict")
def predict(input: TextInput):
    result = predict_single(input.text, input.model_name)
    if result is None:
        return {"error": f"Model '{input.model_name}' not supported."}
    return {
        "input_text": input.text,
        "model_used": input.model_name,
        "predictions": result
    }

# 🚀 Batch texts predict
@app.post("/batch_predict")
def batch_predict(input: BatchInput):
    batch_results = []
    for text in input.texts:
        result = predict_single(text, input.model_name)
        if result is None:
            return {"error": f"Model '{input.model_name}' not supported."}
        batch_results.append({
            "input_text": text,
            "predictions": result
        })
    return {
        "model_used": input.model_name,
        "results": batch_results
    }



# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import math
# import os

# app = FastAPI()

# # model_path = "../models/saved_model_roberta_toxic"
# model_path = os.path.join(os.getcwd(), "models", "saved_model_roberta_toxic")

# tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
# model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# class TextInput(BaseModel):
#     text: str

# @app.post("/predict")
# def predict(input: TextInput):
#     inputs = tokenizer(input.text, return_tensors="pt", padding=True, truncation=True)
    
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         probs = torch.sigmoid(logits).squeeze().tolist()
    
#     # Clean NaNs/Infs
#     prediction = {}
#     for label, prob in zip(labels, probs):
#         if not math.isnan(prob) and not math.isinf(prob):
#             if prob >= 0.5:  # ✅ Only include labels above threshold
#                 prediction[label] = float(prob)
    
#     return {"input_text": input.text, "predictions": prediction}
