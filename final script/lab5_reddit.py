import sys
import re
import time
import threading
import hashlib
from datetime import datetime
from io import BytesIO
import warnings

import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mysql.connector

warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'reddit_lab5'
}

# Source: https://www.reddit.com/explore/29m4k39/technology/
SUBREDDITS = [
    'technology',
    'tech',
    'gadgets',
    'programming',
    'computerscience',
    'artificial',
    'MachineLearning',
    'cybersecurity',
    'netsec',
    'hardware',
    'software',
    'linux',
    'apple',
    'android',
    'gaming',
]

NUM_CLUSTERS = 5
EMBEDDING_DIM = 100
REQUEST_SIZE = 100

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
    'if', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'once', 'can', 'get', 'got', 'like', 'even',
    'new', 'want', 'use', 'used', 'using', 'make', 'made', 'know', 'take', 'see', 'come',
    'think', 'look', 'going', 'way', 'well', 'back', 'much', 'because', 'good', 'give',
    'dont', 'im', 'ive', 'youre', 'thats', 'theyre', 'weve', 'hes', 'shes', 'lets'
}


class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.close()
        conn.close()
        self.conn = mysql.connector.connect(**DB_CONFIG)

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                post_id VARCHAR(255) UNIQUE,
                subreddit VARCHAR(100),
                title TEXT,
                content TEXT,
                author_hash VARCHAR(64),
                timestamp DATETIME,
                url TEXT,
                likes VARCHAR(20),
                comments INT,
                image_text TEXT,
                keywords TEXT,
                topics TEXT,
                embedding BLOB,
                cluster_id INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        try:
            cursor.execute('ALTER TABLE posts ADD COLUMN subreddit VARCHAR(100) AFTER post_id')
        except:
            pass
        self.conn.commit()

    def insert_post(self, post_data):
        cursor = self.conn.cursor()
        embedding_bytes = post_data['embedding'].tobytes() if post_data['embedding'] is not None else None
        cursor.execute('''
            INSERT INTO posts (post_id, subreddit, title, content, author_hash, timestamp, url, likes, comments,
                             image_text, keywords, topics, embedding, cluster_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                subreddit=VALUES(subreddit), title=VALUES(title), content=VALUES(content), likes=VALUES(likes),
                comments=VALUES(comments), keywords=VALUES(keywords), topics=VALUES(topics),
                embedding=VALUES(embedding), cluster_id=VALUES(cluster_id)
        ''', (
            post_data['post_id'], post_data.get('subreddit', ''), post_data['title'], post_data['content'],
            post_data['author_hash'], post_data['timestamp'], post_data['url'],
            post_data['likes'], post_data['comments'], post_data['image_text'],
            post_data['keywords'], post_data['topics'], embedding_bytes, post_data['cluster_id']
        ))
        self.conn.commit()

    def get_all_posts(self):
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM posts')
        posts = cursor.fetchall()
        for post in posts:
            if post['embedding']:
                blob = post['embedding']
                if len(blob) == EMBEDDING_DIM * 4:
                    post['embedding'] = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
                else:
                    post['embedding'] = np.frombuffer(blob, dtype=np.float64)
        return posts

    def get_posts_by_cluster(self, cluster_id):
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM posts WHERE cluster_id = %s', (int(cluster_id),))
        return cursor.fetchall()

    def close(self):
        if self.conn:
            self.conn.close()


class RedditScraper:

 # Politeness   
    BATCH_SIZE = 100
    MAX_PER_LISTING = 1000
    REQUEST_TIMEOUT = 30
    DELAY_BETWEEN_REQUESTS = 3
    DELAY_BETWEEN_SUBREDDITS = 5
    
    def __init__(self, subreddits=None):
        self.subreddits = subreddits or SUBREDDITS
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })
        self._init_session()

    def _init_session(self):
        print("  Initializing session...")
        try:
            resp = self.session.get('https://old.reddit.com/', timeout=self.REQUEST_TIMEOUT)
            print(f"  Session initialized (status: {resp.status_code})")
            time.sleep(1)
        except Exception as e:
            print(f"  Session init failed: {e}")

    def scrape_posts(self, num_posts):
        print(f"Fetching {num_posts} posts from {len(self.subreddits)} subreddits...")
        print(f"  Subreddits: {', '.join(self.subreddits)}")
        
        posts_per_subreddit = num_posts // len(self.subreddits)
        remainder = num_posts % len(self.subreddits)
        
        print(f"  Distribution: ~{posts_per_subreddit} posts per subreddit")
        
        all_posts = []
        start_time = time.time()
        
        for i, subreddit in enumerate(self.subreddits):
            target = posts_per_subreddit + (1 if i < remainder else 0)
            
            if len(all_posts) >= num_posts:
                break
            
            remaining_needed = num_posts - len(all_posts)
            target = min(target, remaining_needed)
            
            if target <= 0:
                break
            
            print(f"\n  [{i+1}/{len(self.subreddits)}] Scraping r/{subreddit} (target: {target} posts)...")
            
            posts = self._scrape_subreddit_json(subreddit, target)
            if not posts:
                print(f"    JSON failed, trying HTML...")
                posts = self._scrape_subreddit_html(subreddit, target)
            
            for post in posts:
                post['subreddit'] = subreddit
            
            all_posts.extend(posts)
            print(f"    Got {len(posts)} posts from r/{subreddit}, total: {len(all_posts)}")
            
            if i < len(self.subreddits) - 1 and len(all_posts) < num_posts:
                print(f"    Waiting {self.DELAY_BETWEEN_SUBREDDITS}s before next subreddit...")
                time.sleep(self.DELAY_BETWEEN_SUBREDDITS)
        
        elapsed = time.time() - start_time
        print(f"\n  Total: {len(all_posts)} posts from {len(self.subreddits)} subreddits in {elapsed:.1f}s")
        return all_posts[:num_posts]

    def _scrape_subreddit_json(self, subreddit, num_posts):
        posts = []
        after = None
        consecutive_errors = 0
        max_consecutive_errors = 3
        total_requests = 0
        
        while len(posts) < num_posts:
            url = f'https://old.reddit.com/r/{subreddit}.json'
            params = {'limit': self.BATCH_SIZE, 'raw_json': 1}
            if after:
                params['after'] = after
            
            total_requests += 1
            print(f"      Request #{total_requests}: fetching posts {len(posts)+1}-{min(len(posts)+self.BATCH_SIZE, num_posts)}")
            
            try:
                resp = self.session.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
            except requests.exceptions.Timeout:
                print(f"        Timeout, retrying...")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(self.DELAY_BETWEEN_REQUESTS * 2)
                continue
            except requests.exceptions.RequestException as e:
                print(f"        Request error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(self.DELAY_BETWEEN_REQUESTS)
                continue
            
            if resp.status_code == 429:
                wait_time = min(60 * (2 ** consecutive_errors), 300)
                print(f"        Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                consecutive_errors += 1
                continue
            
            if resp.status_code != 200:
                print(f"        Error: Status {resp.status_code}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(self.DELAY_BETWEEN_REQUESTS)
                continue
            
            consecutive_errors = 0
            
            try:
                data = resp.json()
            except Exception as e:
                print(f"        JSON parse error: {e}")
                break
            
            children = data.get('data', {}).get('children', [])
            if not children:
                print(f"        No more posts available in r/{subreddit}")
                break
            
            batch_count = 0
            for child in children:
                post_data = child.get('data', {})
                if post_data.get('stickied') or post_data.get('promoted'):
                    continue
                
                post = {
                    'post_id': post_data.get('name', ''),
                    'title': post_data.get('title', ''),
                    'content': post_data.get('selftext', ''),
                    'author': post_data.get('author', '[deleted]'),
                    'timestamp': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat() if post_data.get('created_utc') else None,
                    'url': post_data.get('url', ''),
                    'likes': str(post_data.get('score', 0)),
                    'comments': post_data.get('num_comments', 0),
                    'image_url': post_data.get('thumbnail') if post_data.get('thumbnail', '').startswith('http') else None
                }
                posts.append(post)
                batch_count += 1
                
                if len(posts) >= num_posts:
                    break
            
            print(f"        Got {batch_count} posts, total from r/{subreddit}: {len(posts)}")
            
            after = data.get('data', {}).get('after')
            if not after:
                break
            
            time.sleep(self.DELAY_BETWEEN_REQUESTS)
        
        return posts

    def _scrape_subreddit_html(self, subreddit, num_posts):
        posts = []
        url = f'https://old.reddit.com/r/{subreddit}/'
        consecutive_errors = 0
        max_consecutive_errors = 3
        total_requests = 0
        
        while len(posts) < num_posts:
            total_requests += 1
            print(f"      HTML Request #{total_requests}: fetching from post {len(posts)+1}")
            
            try:
                page = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            except requests.exceptions.Timeout:
                print(f"        Timeout, retrying...")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(self.DELAY_BETWEEN_REQUESTS * 2)
                continue
            except requests.exceptions.RequestException as e:
                print(f"        Request error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(self.DELAY_BETWEEN_REQUESTS)
                continue
            
            if page.status_code == 429:
                wait_time = min(60 * (2 ** consecutive_errors), 300)
                print(f"        Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                consecutive_errors += 1
                continue
            
            if page.status_code != 200:
                print(f"        Error: Status {page.status_code}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(self.DELAY_BETWEEN_REQUESTS)
                continue
            
            consecutive_errors = 0
            soup = BeautifulSoup(page.text, 'html.parser')
            things = soup.find_all('div', class_='thing')
            
            if not things:
                print(f"        No posts found in r/{subreddit}")
                break
            
            batch_count = 0
            for thing in things:
                if len(posts) >= num_posts:
                    break
                
                if 'promoted' in thing.get('class', []):
                    continue
                if thing.get('data-promoted'):
                    continue
                
                post = self._extract_post(thing)
                if post:
                    posts.append(post)
                    batch_count += 1
            
            print(f"        Got {batch_count} posts, total from r/{subreddit}: {len(posts)}")
            
            next_button = soup.find('span', class_='next-button')
            if not next_button:
                break
            
            next_link = next_button.find('a')
            if not next_link or not next_link.get('href'):
                break
            
            url = next_link['href']
            time.sleep(self.DELAY_BETWEEN_REQUESTS)
        
        return posts

    def _extract_post(self, thing):
        post_id = thing.get('data-fullname', '')
        if not post_id:
            return None
        
        # Title
        title_elem = thing.find('p', class_='title')
        title = ''
        if title_elem:
            title_link = title_elem.find('a', class_='title')
            if title_link:
                title = title_link.text.strip()
        
        # Author
        author_elem = thing.find('a', class_='author')
        author = author_elem.text.strip() if author_elem else '[deleted]'
        
        # Likes
        likes_elem = thing.find('div', class_='score')
        likes = '0'
        if likes_elem:
            likes = likes_elem.text.strip()
            if likes == '•':
                likes = 'None'
        
        # Comments
        comments_elem = thing.find('a', class_='comments')
        comments = 0
        if comments_elem:
            comments_text = comments_elem.text.strip().split()[0]
            if comments_text.isdigit():
                comments = int(comments_text)
        
        # Timestamp
        time_elem = thing.find('time')
        timestamp = time_elem.get('datetime') if time_elem else None
        
        # URL
        url = thing.get('data-url', '')
        if not url:
            permalink = thing.get('data-permalink', '')
            if permalink:
                url = 'https://old.reddit.com' + permalink
        
        # Content from expando (self posts)
        content = ''
        expando = thing.find('div', class_='expando')
        if expando:
            md = expando.find('div', class_='md')
            if md:
                content = md.get_text(strip=True)
        
        # Image URL for OCR
        image_url = None
        thumb = thing.find('a', class_='thumbnail')
        if thumb:
            img = thumb.find('img')
            if img and img.get('src', '').startswith('http'):
                image_url = img['src']
        
        return {
            'post_id': post_id,
            'title': title,
            'content': content,
            'author': author,
            'timestamp': timestamp,
            'url': url,
            'likes': likes,
            'comments': comments,
            'image_url': image_url
        }


class DataPreprocessor:
    def preprocess(self, post):
        processed = post.copy()
        processed['title'] = self._clean_text(post.get('title', ''))
        processed['content'] = self._clean_text(post.get('content', ''))
        processed['author_hash'] = self._hash_username(post.get('author', ''))
        processed['timestamp'] = self._parse_timestamp(post.get('timestamp'))
        processed['image_text'] = self._extract_image_text(post.get('image_url'))
        
        full_text = f"{processed['title']} {processed['content']} {processed['image_text']}"
        processed['keywords'] = ','.join(self._extract_keywords(full_text))
        processed['topics'] = ','.join(self._extract_topics(full_text))
        
        return processed

    def _clean_text(self, text):
        if not text:
            return ''
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _hash_username(self, username):
        if not username:
            return hashlib.sha256(b'anonymous').hexdigest()
        return hashlib.sha256(username.encode()).hexdigest()

    def _parse_timestamp(self, ts):
        if not ts:
            return datetime.now()
        if isinstance(ts, datetime):
            return ts
        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z']:
            try:
                return datetime.strptime(ts.replace('+00:00', '+0000'), fmt)
            except ValueError:
                continue
        return datetime.now()

    def _extract_image_text(self, image_url):
        if not image_url or not OCR_AVAILABLE:
            return ''
        try:
            response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(response.content))
            text = pytesseract.image_to_string(img)
            return self._clean_text(text)
        except Exception:
            return ''

    def _extract_keywords(self, text, top_n=10):
        if not text:
            return []
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        words = [w for w in words if w not in STOP_WORDS]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_n]]

    def _extract_topics(self, text):
        tech_topics = {
            'ai': ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'neural', 'gpt', 'llm', 'chatgpt'],
            'security': ['security', 'hack', 'breach', 'cyber', 'privacy', 'vulnerability', 'malware'],
            'mobile': ['phone', 'android', 'ios', 'iphone', 'mobile', 'app', 'smartphone'],
            'hardware': ['cpu', 'gpu', 'chip', 'processor', 'hardware', 'computer', 'laptop'],
            'software': ['software', 'program', 'code', 'developer', 'programming', 'open source'],
            'web': ['web', 'browser', 'internet', 'website', 'online', 'chrome', 'firefox'],
            'gaming': ['game', 'gaming', 'console', 'playstation', 'xbox', 'nintendo', 'steam']
        }
        text_lower = text.lower()
        found = [topic for topic, kws in tech_topics.items() if any(kw in text_lower for kw in kws)]
        return found if found else ['general']


class DocumentEmbedder:
    def __init__(self, max_features=EMBEDDING_DIM):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.fitted = False

    def fit(self, documents):
        docs = [d for d in documents if d and d.strip()]
        if not docs:
            return
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


class MessageClusterer:
    def __init__(self, n_clusters=NUM_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

    def fit(self, embeddings):
        n = min(self.n_clusters, len(embeddings))
        if n < 2:
            return np.zeros(len(embeddings), dtype=int)
        embeddings = np.asarray(embeddings, dtype=np.float64)
        self.kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        return self.kmeans.fit_predict(embeddings)

    def predict(self, embedding):
        if self.kmeans is None:
            return 0
        embedding = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        return int(self.kmeans.predict(embedding)[0])

    def get_cluster_keywords(self, texts, labels, top_n=5):
        cluster_keywords = {}
        for cid in set(labels):
            cluster_texts = [t for t, l in zip(texts, labels) if l == cid]
            if not cluster_texts:
                cluster_keywords[cid] = []
                continue
            try:
                tfidf_matrix = self.tfidf.fit_transform(cluster_texts)
                feature_names = self.tfidf.get_feature_names_out()
                mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
                top_indices = mean_tfidf.argsort()[-top_n:][::-1]
                cluster_keywords[cid] = [feature_names[i] for i in top_indices]
            except Exception:
                cluster_keywords[cid] = []
        return cluster_keywords

    def visualize(self, embeddings, labels, keywords):
        if len(embeddings) < 2:
            print("Not enough data to visualize")
            return
        
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
        
        for i in set(labels):
            mask = labels == i
            if mask.any():
                centroid = reduced[mask].mean(axis=0)
                kw_text = ', '.join(keywords.get(i, [])[:3])
                plt.annotate(f'C{i}: {kw_text}', centroid, fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'Message Clusters (K={len(set(labels))})')
        plt.tight_layout()
        plt.savefig('clusters.png', dpi=150)
        plt.close()
        print("Visualization saved to clusters.png")


class RedditPipeline:
    def __init__(self):
        self.db = DatabaseManager()
        self.scraper = RedditScraper()
        self.preprocessor = DataPreprocessor()
        self.embedder = DocumentEmbedder()
        self.clusterer = MessageClusterer()
        self.running = False
        self.update_thread = None

    def run_pipeline(self, num_posts=100):
        print("=" * 50)
        print(f"[{datetime.now()}] Starting pipeline...")
        
        print("Fetching data...")
        raw_posts = self.scraper.scrape_posts(num_posts)
        print(f"Fetched {len(raw_posts)} posts")
        
        if not raw_posts:
            print("No posts fetched. Skipping pipeline.")
            return []
        
        print("Preprocessing data...")
        processed_posts = [self.preprocessor.preprocess(p) for p in raw_posts]
        
        texts = [f"{p['title']} {p['content']}" for p in processed_posts]
        
        print("Training embeddings...")
        self.embedder.fit(texts)
        
        print("Generating embeddings...")
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
                    if not self.running:
                        break
                    time.sleep(1)
                if self.running:
                    print(f"\n[{datetime.now()}] Running scheduled update...")
                    self.run_pipeline(100)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        print(f"Automation started. Updates every {interval_minutes} minutes.")

    def stop_automation(self):
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)

    def query_cluster(self, query_text):
        processed = self.preprocessor._clean_text(query_text)
        embedding = self.embedder.get_embedding(processed)
        
        if self.clusterer.kmeans is None:
            posts = self.db.get_all_posts()
            if posts:
                embeddings = np.array([p['embedding'] for p in posts if p['embedding'] is not None], dtype=np.float64)
                if len(embeddings) > 0:
                    self.clusterer.fit(embeddings)
        
        cluster_id = self.clusterer.predict(embedding)
        posts = self.db.get_posts_by_cluster(cluster_id)
        
        print(f"\nQuery: '{query_text}'")
        print(f"Matched Cluster: {cluster_id}")
        print(f"Found {len(posts)} related posts:\n")
        
        for i, post in enumerate(posts[:10]):
            subreddit = post.get('subreddit', 'unknown')
            print(f"  {i+1}. [r/{subreddit}] {post['title'][:70]}")
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python lab5_reddit.py <interval_minutes>")
        print("Example: python lab5_reddit.py 5")
        sys.exit(1)
    
    interval = int(sys.argv[1])
    pipeline = RedditPipeline()
    
    pipeline.run_pipeline(REQUEST_SIZE)
    pipeline.start_automation(interval)
    
    try:
        pipeline.interactive_mode()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pipeline.close()


if __name__ == '__main__':
    main()
