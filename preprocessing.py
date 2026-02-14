# preprocessing.py
import re
import hashlib
from datetime import datetime
from io import BytesIO
import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings

from config import STOP_WORDS


warnings.filterwarnings('ignore', category = MarkupResemblesLocatorWarning)

try:
    import pytesseract
    from PIL import Image


    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


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
        if not text: return ''
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
        if not ts: return datetime.now()
        if isinstance(ts, datetime): return ts
        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z']:
            try:
                return datetime.strptime(ts.replace('+00:00', '+0000'), fmt)
            except ValueError:
                continue
        return datetime.now()

    def _extract_image_text(self, image_url):
        if not image_url or not OCR_AVAILABLE: return ''
        try:
            response = requests.get(image_url, timeout = 10)
            img = Image.open(BytesIO(response.content))
            text = pytesseract.image_to_string(img)
            return self._clean_text(text)
        except Exception:
            return ''

    def _extract_keywords(self, text, top_n = 10):
        if not text: return []
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        words = [w for w in words if w not in STOP_WORDS]
        freq = {}
        for w in words: freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key = lambda x: x[1], reverse = True)
        return [w for w, _ in sorted_words[:top_n]]

    def _extract_topics(self, text):
        tech_topics = {
            'ai': ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'neural', 'gpt',
                   'llm', 'chatgpt'],
            'security': ['security', 'hack', 'breach', 'cyber', 'privacy', 'vulnerability',
                         'malware'],
            'mobile': ['phone', 'android', 'ios', 'iphone', 'mobile', 'app', 'smartphone'],
            'hardware': ['cpu', 'gpu', 'chip', 'processor', 'hardware', 'computer', 'laptop'],
            'software': ['software', 'program', 'code', 'developer', 'programming', 'open source'],
            'web': ['web', 'browser', 'internet', 'website', 'online', 'chrome', 'firefox'],
            'gaming': ['game', 'gaming', 'console', 'playstation', 'xbox', 'nintendo', 'steam']
        }
        text_lower = text.lower()
        found = [topic for topic, kws in tech_topics.items() if any(kw in text_lower for kw in kws)]
        return found if found else ['general']