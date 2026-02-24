import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime as dt
import re

URL_PATTERN = r"\b(?:https?://|www\.)[^\s<>\']+"

def remove_non_ascii(text: str) -> str:
    return text.encode("ascii", "ignore").decode("ascii")

class DataCollector:
    start_date: str = "2025-01-20"
    end_date: str = "2026-01-19"

    def __init__(self):
        self.url: str = "https://trumpstruth.org/?sort=desc&per_page=1000"

    def run_data_collection(self):
        results = []

        current_url = self.url
        while True:
            r = self.get_request(current_url)
            soup = BeautifulSoup(r.content, 'lxml')
            posts = soup.select('div.statuses div.status')

            for post in posts:
                text = post.select_one('div.status__content')
                poster, post_date = post.select('a.status-info__meta-item')[:2]
                post_date = dt.strptime(post_date.get_text(), "%B %d, %Y, %I:%M %p")
                results.append({
                    'id': post.select_one('a.status__external-link')['href'].split('/')[-1],
                    'date': post_date.strftime("%Y-%m-%d"),
                    'time': post_date.strftime(f"%H:%M"),
                    'poster': poster.get_text().strip(),
                    'text': text.get_text() if text else None
                })
            current_url = soup.select('a.button.button--xsmall')[-1]['href']
            if results[-1]["date"] < self.start_date:
                break

        df = pd.DataFrame(results)
        df.to_csv('raw_data.csv', index=False)

    def get_request(self, url: str, retries=5):
        for i in range(retries):
            r = requests.get(url)
            if r.ok:
                print(f"Collected url: {url}")
                return r
        raise RuntimeError(f"Unable to collect page status code={r.status_code}")


def collect_data():
    collector = DataCollector()
    collector.run_data_collection()

def clean_data():
    df = pd.read_csv('raw_data.csv')
    df = df[df['date'] <= DataCollector.end_date]
    df = df[df['date'] >= DataCollector.start_date]
    df = df.drop_duplicates(subset='id')
    df = df[df['text'].notnull()]
    df['text'] = df['text'].apply(lambda x: re.sub(URL_PATTERN, "", x).strip())
    df['text'] = df['text'].apply(lambda x: " ".join(remove_non_ascii(x).split()))
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['char_length'] = df['text'].apply(lambda x: len(x))
    df = df[df['word_count']>0]
    df.to_csv('clean_data.csv', index=False)


if __name__ == '__main__':
    collect_data()
    clean_data()
