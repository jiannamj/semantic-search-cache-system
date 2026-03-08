import numpy as np
import skfuzzy as fuzz


class FuzzyClusterer:
    """
    Fuzzy C-Means clustering for document embeddings
    """

    def __init__(self, n_clusters=10):

        self.n_clusters = n_clusters
        self.centers = None
        self.membership = None

    def fit(self, embeddings):

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            embeddings.T,
            c=self.n_clusters,
            m=2,
            error=0.005,
            maxiter=1000
        )

        self.centers = cntr
        self.membership = u

        return cntr, u