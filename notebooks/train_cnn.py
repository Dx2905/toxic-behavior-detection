# notebooks/train_cnn.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pickle

# ⚡ Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 📥 Load Data
print("🔵 Loading data...")
data = pd.read_csv("train.csv")
data = data.sample(1000, random_state=42).reset_index(drop=True)  # Sample 1000 rows

# 🧹 Preprocess
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
data['labels'] = data[label_cols].values.tolist()
texts = data['comment_text'].fillna("none").tolist()
labels = data['labels']

# 🧠 Tokenization
from sklearn.feature_extraction.text import TfidfVectorizer

print("🔵 Fitting TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels.tolist())

# 📦 Dataset
class ToxicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 🔀 Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = ToxicDataset(X_train, y_train)
val_dataset = ToxicDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 🛠️ Model
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

input_dim = X.shape[1]
output_dim = 6

model = CNNClassifier(input_dim, output_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🏋️ Train
print("🔵 Starting training...")
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 💾 Save model
os.makedirs("models/saved_model_cnn", exist_ok=True)
torch.save(model.state_dict(), "models/saved_model_cnn/cnn_model.pt")
with open("models/saved_model_cnn/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ CNN model and tokenizer saved to models/saved_model_cnn")
