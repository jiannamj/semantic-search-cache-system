from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Converts text into vector embeddings.
    """

    def __init__(self):

        # Lightweight but powerful embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts):

        # allow single string or list
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts)

        return embeddings