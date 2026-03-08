# Semantic Search Cache System

## Overview

This project implements a lightweight semantic search system on the **20 Newsgroups dataset**.
It combines **vector embeddings, fuzzy clustering, and a semantic cache** to efficiently handle semantically similar queries.

The system is exposed as a **FastAPI service** that supports semantic query processing and caching.

---

## Architecture

User Query
↓
Text Embedding (Sentence Transformers)
↓
Semantic Cache Lookup
↓
If Cache Hit → Return Cached Result
If Cache Miss → Vector Search (FAISS)
↓
Store Result in Cache

---

## Technologies Used

* Python
* FastAPI
* SentenceTransformers
* FAISS (Vector Database)
* Scikit-learn
* Scikit-fuzzy

---

## Components

### 1. Embedding Layer

Documents and queries are converted into vector embeddings using:

```
all-MiniLM-L6-v2
```

This allows semantic similarity comparison between queries and documents.

---

### 2. Vector Database

FAISS is used to store document embeddings and perform fast similarity search.

---

### 3. Fuzzy Clustering

Documents are clustered using **Fuzzy C-Means** so each document can belong to multiple clusters with different probabilities.

Example:

```
Cluster 1 → 0.42
Cluster 2 → 0.31
Cluster 3 → 0.27
```

---

### 4. Semantic Cache

Instead of exact query matching, the cache uses **cosine similarity** between embeddings.

If similarity exceeds a threshold (e.g., 0.80), the cached result is reused.

---

## API Endpoints

### POST /query

Input:

```
{
 "query": "How do rockets launch?"
}
```

Response:

```
{
 "query": "...",
 "cache_hit": true,
 "matched_query": "...",
 "similarity_score": 0.91,
 "result": "...",
 "dominant_cluster": 3
}
```

---

### GET /cache/stats

Returns cache statistics:

```
{
 "total_entries": 10,
 "hit_count": 4,
 "miss_count": 6,
 "hit_rate": 0.4
}
```

---

### DELETE /cache

Clears the cache and resets statistics.

---

## How to Run

Create virtual environment:

```
python -m venv venv
```

Activate:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run API:

```
uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Key Design Decisions

* **SentenceTransformer embeddings** chosen for semantic similarity quality.
* **FAISS** used for efficient vector search.
* **Fuzzy clustering** allows documents to belong to multiple topics.
* **Semantic caching** reduces recomputation for similar queries.

---

## Future Improvements

* Cluster-aware cache lookup
* Distributed vector search
* Dynamic similarity thresholds
* Query result ranking improvements
