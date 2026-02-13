# scraper.py
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from config import SUBREDDITS

class RedditScraper:
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
        posts_per_subreddit = num_posts // len(self.subreddits)
        remainder = num_posts % len(self.subreddits)
        
        all_posts = []
        start_time = time.time()
        
        for i, subreddit in enumerate(self.subreddits):
            target = posts_per_subreddit + (1 if i < remainder else 0)
            if len(all_posts) >= num_posts or target <= 0:
                break
            
            target = min(target, num_posts - len(all_posts))
            print(f"\n  [{i+1}/{len(self.subreddits)}] Scraping r/{subreddit} (target: {target} posts)...")
            
            posts = self._scrape_subreddit_json(subreddit, target)
            if not posts:
                print(f"    JSON failed, trying HTML...")
                posts = self._scrape_subreddit_html(subreddit, target)
            
            for post in posts:
                post['subreddit'] = subreddit
            
            all_posts.extend(posts)
            
            if i < len(self.subreddits) - 1 and len(all_posts) < num_posts:
                time.sleep(self.DELAY_BETWEEN_SUBREDDITS)
        
        elapsed = time.time() - start_time
        print(f"\n  Total: {len(all_posts)} posts from {len(self.subreddits)} subreddits in {elapsed:.1f}s")
        return all_posts[:num_posts]

    def _scrape_subreddit_json(self, subreddit, num_posts):
        posts = []
        after = None
        consecutive_errors = 0
        
        while len(posts) < num_posts:
            url = f'https://old.reddit.com/r/{subreddit}.json'
            params = {'limit': self.BATCH_SIZE, 'raw_json': 1}
            if after:
                params['after'] = after
                
            try:
                resp = self.session.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
                if resp.status_code == 429:
                    time.sleep(min(60 * (2 ** consecutive_errors), 300))
                    consecutive_errors += 1
                    continue
                if resp.status_code != 200:
                    break
                
                data = resp.json()
                children = data.get('data', {}).get('children', [])
                if not children:
                    break
                
                for child in children:
                    post_data = child.get('data', {})
                    if post_data.get('stickied') or post_data.get('promoted'):
                        continue
                        
                    posts.append({
                        'post_id': post_data.get('name', ''),
                        'title': post_data.get('title', ''),
                        'content': post_data.get('selftext', ''),
                        'author': post_data.get('author', '[deleted]'),
                        'timestamp': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat() if post_data.get('created_utc') else None,
                        'url': post_data.get('url', ''),
                        'likes': str(post_data.get('score', 0)),
                        'comments': post_data.get('num_comments', 0),
                        'image_url': post_data.get('thumbnail') if post_data.get('thumbnail', '').startswith('http') else None
                    })
                    if len(posts) >= num_posts:
                        break
                        
                after = data.get('data', {}).get('after')
                if not after:
                    break
                time.sleep(self.DELAY_BETWEEN_REQUESTS)
                consecutive_errors = 0
                
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    break
                time.sleep(self.DELAY_BETWEEN_REQUESTS)
                
        return posts

    def _scrape_subreddit_html(self, subreddit, num_posts):
        posts = []
        url = f'https://old.reddit.com/r/{subreddit}/'
        consecutive_errors = 0
        
        while len(posts) < num_posts:
            try:
                page = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
                if page.status_code == 429:
                    time.sleep(min(60 * (2 ** consecutive_errors), 300))
                    consecutive_errors += 1
                    continue
                if page.status_code != 200:
                    break
                    
                soup = BeautifulSoup(page.text, 'html.parser')
                things = soup.find_all('div', class_='thing')
                
                if not things:
                    break
                    
                for thing in things:
                    if len(posts) >= num_posts:
                        break
                    if 'promoted' in thing.get('class', []) or thing.get('data-promoted'):
                        continue
                        
                    post = self._extract_post(thing)
                    if post:
                        posts.append(post)
                
                next_button = soup.find('span', class_='next-button')
                if not next_button or not next_button.find('a'):
                    break
                
                url = next_button.find('a')['href']
                time.sleep(self.DELAY_BETWEEN_REQUESTS)
                consecutive_errors = 0
                
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    break
                time.sleep(self.DELAY_BETWEEN_REQUESTS)
                
        return posts

    def _extract_post(self, thing):
        post_id = thing.get('data-fullname', '')
        if not post_id: return None
        
        title_elem = thing.find('p', class_='title')
        title = title_elem.find('a', class_='title').text.strip() if title_elem and title_elem.find('a', class_='title') else ''
        
        author_elem = thing.find('a', class_='author')
        author = author_elem.text.strip() if author_elem else '[deleted]'
        
        likes_elem = thing.find('div', class_='score')
        likes = likes_elem.text.strip() if likes_elem else '0'
        if likes == '•': likes = 'None'
        
        comments_elem = thing.find('a', class_='comments')
        comments = 0
        if comments_elem and comments_elem.text.strip().split()[0].isdigit():
            comments = int(comments_elem.text.strip().split()[0])
            
        time_elem = thing.find('time')
        timestamp = time_elem.get('datetime') if time_elem else None
        
        url = thing.get('data-url', '')
        if not url and thing.get('data-permalink'):
            url = 'https://old.reddit.com' + thing.get('data-permalink')
            
        content = ''
        expando = thing.find('div', class_='expando')
        if expando and expando.find('div', class_='md'):
            content = expando.find('div', class_='md').get_text(strip=True)
            
        image_url = None
        thumb = thing.find('a', class_='thumbnail')
        if thumb and thumb.find('img') and thumb.find('img').get('src', '').startswith('http'):
            image_url = thumb.find('img')['src']
            
        return {
            'post_id': post_id, 'title': title, 'content': content,
            'author': author, 'timestamp': timestamp, 'url': url,
            'likes': likes, 'comments': comments, 'image_url': image_url
        }