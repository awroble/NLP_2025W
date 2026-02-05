import requests
from bs4 import BeautifulSoup
import json
import os
import sys
import math
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import LLMFactory

def get_random_page_content():
    url = "https://en.wikipedia.org/wiki/Special:Random"
    headers = {
        'User-Agent': 'MyRandomPageScript/1.0 (contact: your_email@example.com)'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title_element = soup.find(id="firstHeading")
        if not title_element:
            return None, None, None
        title = title_element.text
        
        content_div = soup.find(id="mw-content-text")
        if not content_div:
            return title, response.url, ""
        
        paragraphs = content_div.find_all('p')
        text_content = "\n\n".join([p.get_text() for p in paragraphs if p.get_text(strip=True)])
        
        return title, response.url, text_content[:4000]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return None, None, None

def generate_q_and_a_batch(articles_batch, model):
    """
    Generates Q&A for a specific batch of articles.
    """
    n = len(articles_batch)
    article_contents = [f"Article {i+1}:\nTitle: {a['title']}\nContent: {a['content']}" for i, a in enumerate(articles_batch)]
    
    prompt = f"""Here are {n} articles from Wikipedia. For each article, please generate one clear and answerable question based on the provided content, and provide a concise answer to that question.

Return your response as a single JSON object. The object should have a key "qa_pairs", which is a list of objects. Each object should have a "question" and an "answer" key.

IMPORTANT: You must return exactly {n} pairs, one for each article in the order provided.

Example format:
{{
  "qa_pairs": [
    {{
      "question": "Question for article 1?",
      "answer": "Answer for article 1."
    }},
    {{
      "question": "Question for article 2?",
      "answer": "Answer for article 2."
    }}
  ]
}}

Here are the articles:

---
""" + "\n\n---\n".join(article_contents)

    try:
        response_text = model.generate(prompt)
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1:
            raise ValueError("No JSON object found in response")
            
        json_part = response_text[start_idx:end_idx]
        data = json.loads(json_part)
        qa_pairs = data.get("qa_pairs", [])
        
        if len(qa_pairs) != n:
            print(f"Warning: Batch size mismatch. Expected {n}, got {len(qa_pairs)}. Padding with blanks.")
            if len(qa_pairs) < n:
                qa_pairs.extend([{"question": "Error generation", "answer": "Error"}] * (n - len(qa_pairs)))
            else:
                qa_pairs = qa_pairs[:n]
                
        return qa_pairs

    except Exception as e:
        print(f"Error processing batch: {e}")
        return [{"question": "Error", "answer": "Error"}] * n

def main():
    # --- Configuration ---
    N_ARTICLES = 100
    BATCH_SIZE = 10
    MODEL_NAME = "gpt-5-mini"
    OUTPUT_FILE = "wiki_questions_factual.json"
    # ---------------------

    print(f"Fetching {N_ARTICLES} random Wikipedia articles...")
    articles = []
    
    with tqdm(total=N_ARTICLES, desc="Fetching articles") as pbar:
        while len(articles) < N_ARTICLES:
            title, url, content = get_random_page_content()
            if title and content and len(content) > 200:
                articles.append({"title": title, "url": url, "content": content})
                pbar.update(1)

    print(f"Loading model: {MODEL_NAME}")
    try:
        model = LLMFactory.create(MODEL_NAME)
    except Exception as e:
        print(f"Failed to load model {MODEL_NAME}: {e}")
        return

    print(f"Generating questions in batches of {BATCH_SIZE}...")
    
    all_results = []
    
    num_batches = math.ceil(len(articles) / BATCH_SIZE)
    
    prompts = []
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        current_batch = articles[start_idx:end_idx]
        n = len(current_batch)
        article_contents = [f"Article {i+1}:\nTitle: {a['title']}\nContent: {a['content']}" for i, a in enumerate(current_batch)]
        
        prompt = f"""Here are {n} articles from Wikipedia. For each article, please generate one clear and answerable question based on the provided content, and provide a concise answer to that question.

Return your response as a single JSON object. The object should have a key "qa_pairs", which is a list of objects. Each object should have a "question" and an "answer" key.

IMPORTANT: You must return exactly {n} pairs, one for each article in the order provided.

Example format:
{{
  "qa_pairs": [
    {{
      "question": "Question for article 1?",
      "answer": "Answer for article 1."
    }},
    {{
      "question": "Question for article 2?",
      "answer": "Answer for article 2."
    }}
  ]
}}

Here are the articles:

---
""" + "\n\n---\n".join(article_contents)
        prompts.append(prompt)

    batch_responses = model.generate_batch(prompts)

    for i, response_text in enumerate(batch_responses):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        current_batch = articles[start_idx:end_idx]
        
        try:
            start_json_idx = response_text.find('{')
            end_json_idx = response_text.rfind('}') + 1
            
            if start_json_idx == -1:
                raise ValueError("No JSON object found in response")
                
            json_part = response_text[start_json_idx:end_json_idx]
            data = json.loads(json_part)
            batch_qa_pairs = data.get("qa_pairs", [])
            
            if len(batch_qa_pairs) != len(current_batch):
                print(f"Warning: Batch size mismatch. Expected {len(current_batch)}, got {len(batch_qa_pairs)}. Padding with blanks.")
                if len(batch_qa_pairs) < len(current_batch):
                    batch_qa_pairs.extend([{"question": "Error generation", "answer": "Error"}] * (len(current_batch) - len(batch_qa_pairs)))
                else:
                    batch_qa_pairs = batch_qa_pairs[:len(current_batch)]
        except Exception as e:
            print(f"Error processing batch response: {e}")
            batch_qa_pairs = [{"question": "Error", "answer": "Error"}] * len(current_batch)

        for j, qa in enumerate(batch_qa_pairs):
            global_idx = start_idx + j
            
            if j < len(current_batch):
                article = current_batch[j]
                all_results.append({
                    "id": f"wiki-factual-{global_idx+1}",
                    "category": "wiki_factual",
                    "prompt": qa.get("question", ""),
                    "expected_response": qa.get("answer", ""),
                    "source_article": {
                        "title": article["title"],
                        "url": article["url"],
                    }
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_results)} questions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()