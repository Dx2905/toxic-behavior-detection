# app/cnn_predict.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import os

app = FastAPI()

# 🛠️ CNN model class (same as used during training)
class CNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 100, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(100, output_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out

# 🔵 Load model and vectorizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(os.getcwd(), "models", "saved_model_cnn", "cnn_model.pt")
vectorizer_path = os.path.join(os.getcwd(), "models", "saved_model_cnn", "tfidf_vectorizer.pkl")

model = CNNClassifier(input_dim=5000, output_dim=6)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 📥 Input
class TextInput(BaseModel):
    text: str

@app.post("/predict-cnn")
def predict(input: TextInput):
    text = input.text
    
    # Preprocess
    X = vectorizer.transform([text]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.sigmoid(outputs).squeeze().tolist()
    
    # Prepare output
    prediction = {}
    for label, prob in zip(labels, probs):
        if not isinstance(prob, float):
            continue
        if prob >= 0.5:
            prediction[label] = float(prob)
    
    return {"input_text": input.text, "predictions": prediction}
