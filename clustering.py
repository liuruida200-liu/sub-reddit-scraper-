# clustering.py
import numpy as np
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from config import NUM_CLUSTERS


class MessageClusterer:
    def __init__(self, n_clusters = NUM_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.tfidf = TfidfVectorizer(max_features = 1000, stop_words = 'english')

    def fit(self, embeddings):
        n = min(self.n_clusters, len(embeddings))
        if n < 2:
            return np.zeros(len(embeddings), dtype = int)
        embeddings = np.asarray(embeddings, dtype = np.float64)
        self.kmeans = KMeans(n_clusters = n, random_state = 42, n_init = 10)
        return self.kmeans.fit_predict(embeddings)

    def predict(self, embedding):
        if self.kmeans is None:
            return 0
        embedding = np.asarray(embedding, dtype = np.float64).reshape(1, -1)
        return int(self.kmeans.predict(embedding)[0])

    def get_cluster_keywords(self, texts, labels, top_n = 5):
        cluster_keywords = {}
        for cid in set(labels):
            cluster_texts = [t for t, l in zip(texts, labels) if l == cid]
            if not cluster_texts:
                cluster_keywords[cid] = []
                continue
            try:
                tfidf_matrix = self.tfidf.fit_transform(cluster_texts)
                feature_names = self.tfidf.get_feature_names_out()
                mean_tfidf = np.asarray(tfidf_matrix.mean(axis = 0)).flatten()
                top_indices = mean_tfidf.argsort()[-top_n:][::-1]
                cluster_keywords[cid] = [feature_names[i] for i in top_indices]
            except Exception:
                cluster_keywords[cid] = []
        return cluster_keywords

    def visualize(self, embeddings, labels, keywords):
        if len(embeddings) < 2:
            print("Not enough data to visualize")
            return

        pca = PCA(n_components = 2)
        reduced = pca.fit_transform(embeddings)

        plt.figure(figsize = (12, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c = labels, cmap = 'viridis',
                              alpha = 0.6)

        for i in set(labels):
            mask = labels == i
            if mask.any():
                centroid = reduced[mask].mean(axis = 0)
                kw_text = ', '.join(keywords.get(i, [])[:3])
                plt.annotate(f'C{i}: {kw_text}', centroid, fontsize = 9, fontweight = 'bold',
                             bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.8))

        plt.colorbar(scatter, label = 'Cluster')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'Message Clusters (K={len(set(labels))})')
        plt.tight_layout()
        plt.savefig('clusters.png', dpi = 150)
        plt.close()
        print("Visualization saved to clusters.png")