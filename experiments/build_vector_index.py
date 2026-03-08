import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from app.embeddings.embedder import Embedder
from app.vectordb.vector_store import VectorStore
from experiments.load_dataset import load_data


docs, labels, names = load_data()

print("Total documents:", len(docs))


embedder = Embedder()

print("Generating embeddings...")

embeddings = embedder.embed(docs)

print("Embedding shape:", embeddings.shape)


dimension = embeddings.shape[1]

vector_store = VectorStore(dimension)

vector_store.add_embeddings(embeddings)

print("Vector index built successfully")