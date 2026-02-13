# config.py

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'reddit_lab5'
}

SUBREDDITS = [
    'technology', 'tech', 'gadgets', 'programming', 'computerscience',
    'artificial', 'MachineLearning', 'cybersecurity', 'netsec',
    'hardware', 'software', 'linux', 'apple', 'android', 'gaming',
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