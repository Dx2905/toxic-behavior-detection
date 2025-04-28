from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path where model was saved
model_save_path = "/models/saved_model_roberta_toxic"

# Try loading
print("🔵 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
print("✅ Tokenizer loaded successfully.")

print("🔵 Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
print("✅ Model loaded successfully.")

