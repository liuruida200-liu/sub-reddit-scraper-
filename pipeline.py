# pipeline.py
import time
import threading
from datetime import datetime
import numpy as np

from database import DatabaseManager
from scraper import RedditScraper
from preprocessing import DataPreprocessor
from embedding import DocumentEmbedder
from clustering import MessageClusterer


class RedditPipeline:
    def __init__(self):
        self.db = DatabaseManager()
        self.scraper = RedditScraper()
        self.preprocessor = DataPreprocessor()
        self.embedder = DocumentEmbedder()
        self.clusterer = MessageClusterer()
        self.running = False
        self.update_thread = None

    def run_pipeline(self, num_posts = 100):
        print("=" * 50)
        print(f"[{datetime.now()}] Starting pipeline...")

        print("Fetching data...")
        raw_posts = self.scraper.scrape_posts(num_posts)

        if not raw_posts:
            print("No posts fetched. Skipping pipeline.")
            return []

        print("Preprocessing data...")
        processed_posts = [self.preprocessor.preprocess(p) for p in raw_posts]
        texts = [f"{p['title']} {p['content']}" for p in processed_posts]

        print("Training embeddings...")
        self.embedder.fit(texts)
        embeddings = self.embedder.transform(texts)
        for i, post in enumerate(processed_posts):
            post['embedding'] = embeddings[i]

        print("Clustering messages...")
        labels = self.clusterer.fit(embeddings)
        for i, post in enumerate(processed_posts):
            post['cluster_id'] = int(labels[i])

        keywords = self.clusterer.get_cluster_keywords(texts, labels)
        print("\nCluster Keywords:")
        for cid, kws in sorted(keywords.items()):
            print(f"  Cluster {cid}: {', '.join(kws)}")

        print("\nStoring in database...")
        for post in processed_posts:
            self.db.insert_post(post)

        print("Creating visualization...")
        self.clusterer.visualize(embeddings, labels, keywords)

        print(f"\n[{datetime.now()}] Pipeline complete. Stored {len(processed_posts)} posts.")
        print("=" * 50)
        return processed_posts

    def start_automation(self, interval_minutes):
        self.running = True

        def update_loop():
            while self.running:
                for _ in range(interval_minutes * 60):
                    if not self.running: break
                    time.sleep(1)
                if self.running:
                    print(f"\n[{datetime.now()}] Running scheduled update...")
                    self.run_pipeline(100)

        self.update_thread = threading.Thread(target = update_loop, daemon = True)
        self.update_thread.start()
        print(f"Automation started. Updates every {interval_minutes} minutes.")

    def stop_automation(self):
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout = 5)

    def query_cluster(self, query_text):
        processed = self.preprocessor._clean_text(query_text)
        embedding = self.embedder.get_embedding(processed)

        if self.clusterer.kmeans is None:
            posts = self.db.get_all_posts()
            if posts:
                embeddings = np.array([p['embedding'] for p in posts if p['embedding'] is not None],
                                      dtype = np.float64)
                if len(embeddings) > 0:
                    self.clusterer.fit(embeddings)

        cluster_id = self.clusterer.predict(embedding)
        posts = self.db.get_posts_by_cluster(cluster_id)

        print(f"\nQuery: '{query_text}'")
        print(f"Matched Cluster: {cluster_id}")
        print(f"Found {len(posts)} related posts:\n")

        for i, post in enumerate(posts[:10]):
            subreddit = post.get('subreddit', 'unknown')
            print(f"  {i + 1}. [r/{subreddit}] {post['title'][:70]}")
            if post['content']:
                print(f"     {post['content'][:100]}...")
            print()

        return cluster_id, posts

    def interactive_mode(self):
        print("\n" + "=" * 50)
        print("Interactive Query Mode")
        print("Enter a keyword or message to find related posts.")
        print("Commands: 'quit' to exit, 'refresh' to update data")
        print("=" * 50 + "\n")

        while True:
            try:
                query = input("Query> ").strip()
            except EOFError:
                break
            if query.lower() == 'quit':
                break
            elif query.lower() == 'refresh':
                self.run_pipeline(100)
            elif query:
                self.query_cluster(query)

    def close(self):
        self.stop_automation()
        self.db.close()