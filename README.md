# 👗 Fashion Search Engine 🔍

A content-based image retrieval system for fashion products using **CLIP model** and **Weaviate vector database**.  
This project works like **Google Lens for fashion** — upload a fashion image and get visually similar products from the dataset.

---

## 🚀 Project Overview
- Extract embeddings from fashion product images using **OpenAI’s CLIP model**.
- Store embeddings in a **Weaviate vector database** for similarity search.
- Query test images by computing their embeddings and retrieving top similar products.
- Built using **Myntra Fashion Dataset** (available on Kaggle).
- ⚠️ **Note:** This project has no UI — it is focused only on the ML model pipeline and database integration.

---

## 📂 Repository Structure
```bash
├── try1.py        # First version: Connects to Weaviate and stores embeddings
├── try2.py        # Second version: Alternative code for storing embeddings in Weaviate
├── test.py        # Retrieves predictions for a query image (matches embeddings)
├── requirements.txt  # Python dependencies
└── README.md

```



## ⚡ Workflow

### 1. Embedding Extraction  
- Use **CLIP** to convert images into vector embeddings.  

### 2. Database Storage  
- Store embeddings into **Weaviate (vector DB)** for fast similarity search.  
- Both `try1.py` and `try2.py` contain code for embedding storage and DB connection.  

### 3. Prediction / Search  
- `test.py` takes an input image, generates its embedding, and retrieves similar product embeddings from Weaviate.  

---

## 📊 Dataset
- **Myntra Fashion Dataset (Kaggle):** [Link to Dataset](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset)  
- Dataset contains fashion product images used for training and retrieval.  

---

## 🛠️ Tech Stack
- **Python**  
- **CLIP (OpenAI)** – for image embeddings  
- **Weaviate** – vector database for similarity search  
- **Myntra Dataset (Kaggle)** – fashion product dataset  

---

## 🔧 Installation & Usage

### Clone the repository:
```bash
git clone https://github.com/Dharam13/fashion-search-engine.git
cd fashion-search-engine
