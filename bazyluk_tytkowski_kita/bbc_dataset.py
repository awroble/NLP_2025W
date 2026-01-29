additional_sources = '''
    "https://feeds.npr.org/1004/rss.xml",  # world
    "https://feeds.npr.org/1014/rss.xml",  # politics
    "https://feeds.npr.org/1019/rss.xml",  # tech
'''

import requests
import trafilatura
import feedparser
from datetime import datetime
from urllib.parse import urlparse
import time

# Comprehensive BBC News RSS feeds for broader coverage
RSS_FEEDS = {
    'world': "https://feeds.bbci.co.uk/news/world/rss.xml",
    'uk': "https://feeds.bbci.co.uk/news/uk/rss.xml",
    'business': "https://feeds.bbci.co.uk/news/business/rss.xml",
    'politics': "https://feeds.bbci.co.uk/news/politics/rss.xml",
    'health': "https://feeds.bbci.co.uk/news/health/rss.xml",
    'science_and_environment': "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    'technology': "https://feeds.bbci.co.uk/news/technology/rss.xml",
    'other': "https://feeds.bbci.co.uk/news/rss.xml",
}



def is_likely_article(entry):
    """Filter out non-article items (promos, apps, duplicates)"""
    title = getattr(entry, 'title', '').lower()
    link = getattr(entry, 'link', '').lower()

    # Exclude obvious non-articles
    exclude_patterns = [
        'bbc news app', 'download the app', 'get the bbc',
        'iplayer', 'sounds', 'podcast', 'subscribe'
    ]

    if any(pattern in title or pattern in link for pattern in exclude_patterns):
        return False

    # Prefer entries with both title and description
    return bool(getattr(entry, 'title', ''))

def parse_pubdate(pubdate_str):
    """Parse RSS pubDate like 'Wed, 26 Nov 2025 23:27:17 GMT' to timestamp"""
    if not pubdate_str:
        return 0
    try:
        # RSS standard RFC 822 format: 'Wed, 26 Nov 2025 23:27:17 GMT'
        dt = datetime.strptime(pubdate_str, '%a, %d %b %Y %H:%M:%S %Z')
        return int(time.mktime(dt.timetuple()))
    except ValueError:
        try:
            parsed = feedparser.parse_date(pubdate_str)
            return int(parsed) if parsed else 0
        except:
            return 0

all_articles = []

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print("Fetching BBC News articles from multiple RSS feeds...\n")

for key, feed_url in RSS_FEEDS.items():
    try:
        # Method 1: Use feedparser (recommended, more robust)
        feed = feedparser.parse(feed_url, request_headers=headers)

        for entry in feed.entries:
            if is_likely_article(entry):
                pubdate = parse_pubdate(getattr(entry, 'published', ''))
                all_articles.append({
                    'url': entry.link,
                    'title': entry.title,
                    'pubdate': pubdate,
                    'topic': key
                })

    except Exception as e2:
        print(f"Failed: {feed_url}: {e2}")

# Remove duplicates based on URL and sort by publication date (newest first)
articles = []
seen_urls = set()
for article in sorted(all_articles, key=lambda x: x['pubdate'], reverse=True):
    if article['url'] not in seen_urls:
        articles.append(article)
        seen_urls.add(article['url'])

print(f"\n✅ Found {len(articles)} unique news articles")
print(f"📊 From {len(set(a['topic'] for a in articles))} different feeds\n")

import json

# Load the saved articles
with open('bbc_news_articles.json', 'r') as f:
    full_articles = json.load(f)

print(f"Loaded {len(full_articles)} articles")


used_urls = [article['url'] for article in full_articles]
for i in range(len(articles)):
  url = articles[i]['url']
  if url not in used_urls:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    text = trafilatura.extract(response.text, include_comments=False)
    articles[i]['text'] = " ".join(text.replace("\n", " ").strip().split())
    full_articles.append(articles[i])

with open('bbc_news_articles.json', 'w') as f:
    json.dump(full_articles, f, indent=2)
print(f"Saved {len(full_articles)} articles")