# embedding.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from config import EMBEDDING_DIM

class DocumentEmbedder:
    def __init__(self, max_features=EMBEDDING_DIM):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.fitted = False

    def fit(self, documents):
        docs = [d for d in documents if d and d.strip()]
        if not docs: return
        self.vectorizer.fit(docs)
        self.fitted = True

    def transform(self, documents):
        if not self.fitted:
            return np.zeros((len(documents), EMBEDDING_DIM), dtype=np.float64)
        vectors = self.vectorizer.transform(documents).toarray()
        if vectors.shape[1] < EMBEDDING_DIM:
            padding = np.zeros((vectors.shape[0], EMBEDDING_DIM - vectors.shape[1]))
            vectors = np.hstack([vectors, padding])
        return vectors.astype(np.float64)

    def get_embedding(self, text):
        if not self.fitted:
            return np.zeros(EMBEDDING_DIM, dtype=np.float64)
        vec = self.vectorizer.transform([text]).toarray()[0]
        if len(vec) < EMBEDDING_DIM:
            vec = np.pad(vec, (0, EMBEDDING_DIM - len(vec)))
        return vec.astype(np.float64)