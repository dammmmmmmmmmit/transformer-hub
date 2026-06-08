A comprehensive interactive web app for exploring, comparing, and generating text using major Transformer architectures built with Streamlit and HuggingFace Transformers.

## 📌 What is this?

This app was built as part of my LLM class. It covers the full lifecycle of a story/poem generation system from model analysis to deployment.

The app has 3 sections:

### 📚 Analysis
A structured comparative study of the 3 major Transformer paradigms:
- **Encoder-Only** (BERT, RoBERTa, DistilBERT)
- **Decoder-Only** (GPT-2, GPT-3, GPT-Neo)
- **Encoder-Decoder** (T5, BART, FLAN-T5)

Includes technical comparison tables, quantitative benchmarks (BLEU, ROUGE, Perplexity, SQuAD F1), strengths/weaknesses, application recommendations, and future trends.

### ✍️ Generator
A story and poem generation system powered by 6 open-source models:
- GPT-2 Small / GPT-2 Medium
- GPT-Neo 125M
- T5-Small
- BART-Base
- FLAN-T5-Small

Features:
- Adjustable generation parameters (Temperature, Top-k, Top-p, Max Length, Num Sequences)
- Automatic evaluation metrics (BLEU, ROUGE-1, ROUGE-L, Perplexity)
- Side-by-side model comparison on the same prompt

### 🔬 Architectures
Per-model deep dives with:
- Visual architecture diagrams
- Full mathematical intuition (attention equations, training objectives, scaling laws)
- Key configuration details

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Models:** HuggingFace Transformers
- **Evaluation:** NLTK (BLEU), rouge-score (ROUGE), GPT-2 perplexity
- **Framework:** PyTorch

## 🚀 Run Locally

```bash
git clone https://github.com/dammmmmmmmmmit/transformer-hub.git
cd transformer-hub
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Requirements
transformers==4.40.0
torch==2.2.0
nltk==3.8.1
rouge-score==0.1.2
streamlit>=1.32.0

## 📁 Project Structure
transformer-hub/
├── app.py              # Main Streamlit application
├── requirements.txt    # Dependencies
└── README.md           # This file

## 📖 References
- Vaswani et al. (2017) — Attention Is All You Need
- Devlin et al. (2018) — BERT
- Radford et al. (2019) — GPT-2
- Brown et al. (2020) — GPT-3
- Raffel et al. (2020) — T5
- Lewis et al. (2019) — BART
- Chung et al. (2022) — FLAN-T5