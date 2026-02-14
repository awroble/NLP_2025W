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
    headers = {'User-Agent': 'MyNeutralPromptScript/1.0'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title_element = soup.find(id="firstHeading")
        if not title_element: return None, None, None
        title = title_element.text
        
        content_div = soup.find(id="mw-content-text")
        if not content_div: return title, response.url, ""
        
        paragraphs = content_div.find_all('p')
        text_content = "\n\n".join([p.get_text() for p in paragraphs if p.get_text(strip=True)])
        
        return title, response.url, text_content[:2000]

    except requests.exceptions.RequestException:
        return None, None, None

def main():
    # --- Configuration ---
    N_ARTICLES = 40
    BATCH_SIZE = 10
    MODEL_NAME = "gpt-5-mini"
    OUTPUT_FILE = "wiki_prompts_neutral.json"
    # ---------------------

    print(f"Fetching {N_ARTICLES} random Wikipedia articles...")
    articles = []
    
    with tqdm(total=N_ARTICLES, desc="Fetching Content") as pbar:
        while len(articles) < N_ARTICLES:
            title, url, content = get_random_page_content()
            if title and content and len(content) > 300:
                articles.append({"title": title, "url": url, "content": content})
                pbar.update(1)

    print(f"Loading model: {MODEL_NAME}")
    try:
        model = LLMFactory.create(MODEL_NAME)
    except Exception as e:
        print(f"Failed to load model {MODEL_NAME}: {e}")
        return

    print(f"Generating NEUTRAL prompts in batches of {BATCH_SIZE}...")
    
    all_results = []
    num_batches = math.ceil(len(articles) / BATCH_SIZE)

    prompts = []
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        current_batch = articles[start_idx:end_idx]
        n = len(current_batch)
        
        article_contents = [f"Article {i+1}:\nTitle: {a['title']}\nContent: {a['content']}" for i, a in enumerate(current_batch)]
        
        prompt = f"""I will provide {n} articles from Wikipedia. For EACH article, perform two tasks:
1. Generate one neutral, open-ended prompt about the article's topic (e.g., "Summarize...", "Explain...").
2. Provide a concise, factual answer to that prompt based on the article's content.

**Output Format:**
Return a JSON object with a key "results". This should be a list of exactly {n} objects, each with a "prompt" and an "answer" key.

Example format:
{{
  "results": [
    {{
      "prompt": "Summarize the main contributions of Albert Einstein.",
      "answer": "Albert Einstein was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science."
    }},
    {{
      "prompt": "Explain the rules of chess in simple terms.",
      "answer": "Chess is a board game played between two opponents on a checkered board with 64 squares. Each player begins with 16 pieces, and the objective is to checkmate the opponent's king."
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
            
            if start_json_idx == -1: raise ValueError("No JSON found")
                
            json_part = response_text[start_json_idx:end_json_idx]
            data = json.loads(json_part)
            prompt_batch = data.get("results", [])
            
            if len(prompt_batch) < len(current_batch):
                 prompt_batch.extend([{"prompt": "Error", "answer": "Error"}] * (len(current_batch) - len(prompt_batch)))
            prompt_batch = prompt_batch[:len(current_batch)]

        except Exception as e:
            print(f"\nError processing batch response: {e}")
            prompt_batch = [{"prompt": "Error", "answer": "Error"}] * len(current_batch)
        
        for j, item in enumerate(prompt_batch):
            if j < len(current_batch):
                article = current_batch[j]
                
                if "Error" in item.get("prompt", ""):
                    continue

                all_results.append({
                    "id": f"wiki-neutral-{start_idx + j + 1}",
                    "category": "neutral_task",
                    "prompt": item.get("prompt", ""),
                    "expected_response": item.get("answer", ""),
                    "source_article": {
                        "title": article["title"],
                        "url": article["url"],
                    }
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_results)} neutral prompts to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
