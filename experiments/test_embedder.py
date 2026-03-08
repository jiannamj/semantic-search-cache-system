import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embeddings.embedder import Embedder


embedder = Embedder()

text = "How do rockets launch?"

embedding = embedder.embed(text)

print("Embedding shape:", embedding.shape)
print("First few values:", embedding[0][:10])