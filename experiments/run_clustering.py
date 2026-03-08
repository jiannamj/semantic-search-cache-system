import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embeddings.embedder import Embedder
from app.clustering.fuzzy_cluster import FuzzyClusterer
from experiments.load_dataset import load_data


docs, labels, names = load_data()

print("Total documents:", len(docs))


embedder = Embedder()

print("Generating embeddings...")

embeddings = embedder.embed(docs)

print("Embedding shape:", embeddings.shape)


clusterer = FuzzyClusterer(n_clusters=10)

print("Running fuzzy clustering...")

centers, membership = clusterer.fit(embeddings)

print("Cluster centers shape:", centers.shape)
print("Membership matrix shape:", membership.shape)