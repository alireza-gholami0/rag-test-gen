# RAG Test Scenario Generator (rag-test-gen)

This project is a **Test Scenario Generator** powered by **RAG (Retrieval-Augmented Generation)** using **LangChain**, **HuggingFace embeddings**, and **OpenAI-compatible models**.  
It allows you to upload domain-related PDF files, index them with FAISS, and generate **step-by-step test scenarios** in Persian.

## Features
- Upload and index multiple PDF documents
- Retrieve relevant context using FAISS vector store
- Generate test scenarios with GPT-like models (OpenAI API compatible)
- Supports example test scenarios for better context
- Persian RTL interface built with Streamlit

## Requirements
- Python 3.9+
- Streamlit
- LangChain & related libraries
- FAISS
- HuggingFace Sentence Transformers

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/rag-test-gen.git
cd rag-test-gen
```

Create and activate a virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Start the Streamlit app:
```bash
streamlit run app.py
```

1. Upload one or more PDF documents.
2. Click **"بارگذاری/ایندکس PDF"** to index the documents.
3. Enter your requirements and scenario description.
4. (Optional) Add example scenario description and steps.
5. Click **"تولید سناریو"** to generate a test scenario.

## File Structure
```
ragtestgen/
│── app.py              # Main Streamlit app
│── requirements.txt    # Python dependencies
│── .gitignore          # Git ignore rules
│── README.md           # Project documentation
└── faiss_index/        # Generated FAISS index (after upload)
```
## Author
Created by Alireza Gholami.
