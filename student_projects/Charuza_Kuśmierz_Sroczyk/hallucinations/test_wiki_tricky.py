import requests
from bs4 import BeautifulSoup
import json
import os
import sys
from tqdm import tqdm
import math

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
        return title, response.url, text_content[:5000]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return None, None, None

def generate_tricky_batch(articles_batch, model):
    """
    Generates tricky questions for a small batch of articles.
    """
    formatted_articles = ""
    for i, a in enumerate(articles_batch):
        formatted_articles += f"--- ARTICLE {i+1} ---\nTitle: {a['title']}\nContent Snippet: {a['content'][:2000]}\n\n"

    prompt = f"""You are an expert at creating 'Adversarial' or 'Tricky' benchmark questions.
I will provide {len(articles_batch)} short texts. For EACH text, create exactly one tricky question based on the content.

The goal is to test if an AI model is paying attention or just hallucinating.

**Your Strategy for Tricky Questions:**
1. **The False Premise:** Ask a question that assumes something false is true. (e.g., Text says "Obama was born in Hawaii"; Question: "Which city in Kenya was Obama born in?")
2. **The Entity Swap:** Swap who did what. (e.g., Text says "Steve Jobs founded Apple"; Question: "In what year did Steve Jobs found Microsoft?")
3. **The Date Displacement:** Change the date slightly. (e.g., Text says "Titanic sank in 1912"; Question: "Why did the Titanic sink in 1915?")

**Output Requirement:**
Return a purely JSON object with a key "qa_pairs" containing a list of {len(articles_batch)} objects.
Each object must have:
- "question": The tricky question.
- "answer": The CORRECT factual response that corrects the false premise (refusal to answer the trick).
- "trick_type": One of ["False Premise", "Entity Swap", "Date Displacement", "Impossible Constraint"].

**Input Articles:**
{formatted_articles}

**JSON Output:**
"""

    try:
        response_text = model.generate(prompt)
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")
            
        json_part = response_text[start_idx:end_idx]
        data = json.loads(json_part)
        return data.get("qa_pairs", [])
    except Exception as e:
        print(f"\nError processing batch: {e}")
        return [{"question": "Error generating", "answer": "Error", "trick_type": "Error"}] * len(articles_batch)

def main():
    # --- Configuration ---
    N_ARTICLES = 400
    BATCH_SIZE = 5
    MODEL_NAME = "gpt-5-mini"
    OUTPUT_FILE = "wiki_questions_tricky.json"
    # ---------------------

    print(f"Fetching {N_ARTICLES} random Wikipedia articles...")
    articles = []
    
    with tqdm(total=N_ARTICLES, desc="Fetching Content") as pbar:
        while len(articles) < N_ARTICLES:
            title, url, content = get_random_page_content()
            if title and content and len(content) > 500:
                articles.append({"title": title, "url": url, "content": content})
                pbar.update(1)

    print(f"Loading model: {MODEL_NAME}")
    try:
        model = LLMFactory.create(MODEL_NAME)
    except Exception as e:
        print(f"Failed to load model {MODEL_NAME}: {e}")
        return

    print("Generating TRICKY questions (Adversarial Mode)...")
    
    all_results = []
    
    num_batches = math.ceil(len(articles) / BATCH_SIZE)
    
    prompts = []
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch = articles[start_idx:end_idx]
        
        formatted_articles = ""
        for i_batch, a in enumerate(batch):
            formatted_articles += f"--- ARTICLE {i_batch+1} ---\nTitle: {a['title']}\nContent Snippet: {a['content'][:2000]}\n\n"

        prompt = f"""You are an expert at creating 'Adversarial' or 'Tricky' benchmark questions.
I will provide {len(batch)} short texts. For EACH text, create exactly one tricky question based on the content.

The goal is to test if an AI model is paying attention or just hallucinating.

**Your Strategy for Tricky Questions:**
1. **The False Premise:** Ask a question that assumes something false is true. (e.g., Text says "Obama was born in Hawaii"; Question: "Which city in Kenya was Obama born in?")
2. **The Entity Swap:** Swap who did what. (e.g., Text says "Steve Jobs founded Apple"; Question: "In what year did Steve Jobs found Microsoft?")
3. **The Date Displacement:** Change the date slightly. (e.g., Text says "Titanic sank in 1912"; Question: "Why did the Titanic sink in 1915?")

**Output Requirement:**
Return a purely JSON object with a key "qa_pairs" containing a list of {len(batch)} objects.
Each object must have:
- "question": The tricky question.
- "answer": The CORRECT factual response that corrects the false premise (refusal to answer the trick).
- "trick_type": One of ["False Premise", "Entity Swap", "Date Displacement", "Impossible Constraint"].

**Input Articles:**
{formatted_articles}

**JSON Output:**
"""
        prompts.append(prompt)

    batch_responses = model.generate_batch(prompts)

    for i, response_text in enumerate(batch_responses):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch = articles[start_idx:end_idx]
        
        try:
            start_json_idx = response_text.find('{')
            end_json_idx = response_text.rfind('}') + 1
            if start_json_idx == -1 or end_json_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_part = response_text[start_json_idx:end_json_idx]
            data = json.loads(json_part)
            qa_batch = data.get("qa_pairs", [])
        except Exception as e:
            print(f"\nError processing batch response: {e}")
            qa_batch = [{"question": "Error generating", "answer": "Error", "trick_type": "Error"}] * len(batch)

        for j, qa in enumerate(qa_batch):
            if j < len(batch):
                article = batch[j]
                all_results.append({
                    "id": f"wiki-tricky-{start_idx + j + 1}",
                    "category": "adversarial_fact",
                    "trick_type": qa.get("trick_type", "Unknown"),
                    "prompt": qa.get("question", ""),
                    "expected_response": qa.get("answer", ""),
                    "source_article": {
                        "title": article["title"],
                        "url": article["url"],
                    }
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_results)} tricky questions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()