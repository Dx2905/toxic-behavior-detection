
---
# 🚀 **Toxic Online Behavior Detection and Classification**

## 📌 Project Overview
This project builds a **production-ready AI system** for detecting and classifying **toxic online behavior** like hate speech, cyberbullying, threats, and harassment.  
It leverages **state-of-the-art Transformer models** (RoBERTa, BART), **CNN deep learning**, and **FastAPI deployment** to deliver **scalable** and **real-time toxicity detection**.

By combining **fine-tuned transformer models**, **custom CNNs**, and **batch inference APIs**, this project enables **accurate**, **interpretable**, and **scalable** online moderation solutions.

---

## 🏛 Dataset Overview
- **Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Wikipedia talk page comments)
- **Size:** ~160,000 comments
- **Labels (Multi-label Classification):**
  - **Toxic**
  - **Severe Toxic**
  - **Obscene**
  - **Threat**
  - **Insult**
  - **Identity Hate**

---

## 🔥 Models Implemented

### 🔹 Fine-tuned Transformer Models
| Model | Architecture |
|:------|:-------------|
| **RoBERTa** (`roberta-base`) | Fine-tuned for multi-label toxic classification |
| **BART** (`facebook/bart-base`) | Fine-tuned for multi-label toxic classification |

### 🔹 Deep Learning
| Model | Architecture |
|:------|:-------------|
| **CNN (TF-IDF Based)** | 1D CNN over TF-IDF features for multi-label classification |

---

## ⚙️ Model Training & Inference Pipeline
- **Data Preprocessing:** 
  - Text cleaning and normalization
  - Label binarization
- **Transformers:**
  - Tokenization with HuggingFace Tokenizers
  - Fine-tuning RoBERTa and BART on toxicity detection
- **CNN:**
  - TF-IDF vectorization
  - CNN with convolution + pooling over text features
- **Batch Inference API:**
  - Real-time API serving using **FastAPI**
  - Supports both **single** and **batch** predictions
  - Flexible model selection (`roberta`, `bart`, `cnn`)

---

## 🛠 Project Structure
```
toxic-behavior-classification/
├── app/
│   └── main.py           # FastAPI application for serving models
├── models/
│   ├── saved_model_roberta_toxic/
│   ├── saved_model_bart_toxic/
│   └── saved_model_cnn/
├── notebooks/
│   ├── fine_tune_transformer.py    # RoBERTa fine-tuning script
│   ├── fine_tune_bart.py           # BART fine-tuning script
│   ├── train_cnn.py                # CNN training script (TF-IDF based)
│   └── quick_save_model.py         # Utility for quick model saving
├── Dockerfile             # Dockerfile for containerizing FastAPI app
└── README.md              # Project documentation (this file)
```

---

## 📊 Performance Metrics
| Model      | Macro F1-Score (sample test) |
|:-----------|:-----------------------------|
| **RoBERTa** | High macro F1 (~0.85+) |
| **BART**    | High macro F1 (~0.83+) |
| **CNN (TF-IDF)** | Good baseline (~0.78+) |

---

## 🔧 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Dx2905/Toxic-Behavior-Detection.git
cd Toxic-Behavior-Detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` not provided, install manually:
```bash
pip install torch transformers fastapi uvicorn scikit-learn
```

### 3️⃣ Train Models (Optional if you want fresh training)
```bash
python notebooks/fine_tune_transformer.py
python notebooks/fine_tune_bart.py
python notebooks/train_cnn.py
```

### 4️⃣ Launch FastAPI server (local development)
```bash
uvicorn app.main:app --reload
```
Visit **http://127.0.0.1:8000/docs** to explore the **interactive Swagger UI**! 🚀

---

## 🧪 How to Use APIs
| Endpoint | Description |
|:---------|:------------|
| `/predict` | Predict toxicity for **single input text** |
| `/batch_predict` | Predict toxicity for **batch of input texts** |

- **Input Format:**
  ```json
  {
    "text": "your comment here",
    "model_name": "roberta"
  }
  ```
  or for batch:
  ```json
  {
    "texts": ["comment1", "comment2"],
    "model_name": "bart"
  }
  ```

- **Model options:** `"roberta"`, `"bart"`, (optionally `"cnn"`)

---

## 🧠 Key Learnings
- Fine-tuned **RoBERTa** and **BART** for multi-label toxic classification.
- Built **CNN** model with **TF-IDF embeddings** as an alternative lightweight solution.
- Deployed **FastAPI** for real-time and batch model inference.
- Enabled **dynamic model switching** in API (`roberta`, `bart`, `cnn`).
- Packaged everything inside a **Docker container** for production-ready scaling.

---

## 🛣 Future Improvements
- Integrate full **multi-GPU training** for faster fine-tuning.
- Expand API to serve **model ensemble** (average RoBERTa + BART outputs).
- Create a simple **Streamlit frontend**.
- Implement **monitoring and logging** (Prometheus, Grafana).

---

## 📜 License
This project is licensed under the **MIT License**.  
See the [LICENSE](https://github.com/Dx2905/Toxic-Behavior-Detection/blob/main/LICENSE) file for details.

---
