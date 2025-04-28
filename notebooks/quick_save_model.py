# notebooks/quick_save_model.py

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ Define safe save path
model_save_path = os.path.join(os.getcwd(), "models", "saved_model_roberta_toxic")
os.makedirs(model_save_path, exist_ok=True)

# ✅ Load a basic model and tokenizer (quick sample)
print("\n🔵 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
print("✅ Tokenizer loaded successfully.")

print("\n🔵 Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=6,
    problem_type="multi_label_classification"
)
print("✅ Model loaded successfully.")

# ✅ Save model and tokenizer
print("\n💾 Saving model and tokenizer...")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"✅ Model and tokenizer saved to {model_save_path}")
