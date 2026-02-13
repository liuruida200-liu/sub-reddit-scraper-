# database.py
import mysql.connector
import numpy as np
from config import DB_CONFIG, EMBEDDING_DIM

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