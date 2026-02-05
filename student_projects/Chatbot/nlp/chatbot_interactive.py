#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive RAG Chatbot Using Groq API

This module implements a conversational Retrieval-Augmented Generation (RAG)
system. Users query a knowledge base stored as CSV, which is embedded and
indexed for efficient retrieval. Relevant documents are fetched and used
to augment the prompt before generating responses via Groq API.

Architecture:
  - Sentence-Transformers: Dense vector embeddings of knowledge base
  - FAISS: Fast approximate nearest neighbor search
  - Groq API: Large language model for answer generation

Data Flow:
  1. User enters natural language query
  2. Encode query using embedding model
  3. Search FAISS index for K nearest neighbors
  4. Retrieve Q&A pairs from knowledge base
  5. Construct context prompt with retrieved documents
  6. Query Groq API with context and question
  7. Return answer and source URLs to user

Usage:
    export GROQ_API_KEY='your_api_key'
    python chatbot_rag_groq.py
"""

import os
import sys
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================================
# CONFIGURATION
# =========================================================================

CSV_FILE = "chatbot_csv.csv"
TOP_K = 3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"


# =========================================================================
# RAG CHATBOT IMPLEMENTATION
# =========================================================================

class RAGChatbot:
    """
    Retrieval-Augmented Generation chatbot for knowledge base question-answering.

    This class manages the complete RAG pipeline:
      - Loads knowledge base from CSV
      - Computes and indexes dense vector embeddings
      - Retrieves relevant documents for user queries
      - Generates contextually-grounded answers via LLM
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize RAG chatbot with knowledge base.

        Process:
          1. Load and validate dataframe
          2. Create unified text field for retrieval
          3. Load sentence embedding model
          4. Compute embeddings for all documents
          5. Build FAISS index for fast retrieval
          6. Initialize Groq API client

        Parameters
        ----------
        dataframe : pd.DataFrame
            Knowledge base with columns: Question, Answer, Source_URLs
            (Source_URLs is optional but recommended)

        Raises
        ------
        ValueError
            If required columns are missing from dataframe.
        KeyError
            If embedding model cannot be loaded.
        RuntimeError
            If GROQ_API_KEY is not configured.
        """
        self.df = dataframe.reset_index(drop=True)

        self.texts = (
            self.df["Question"].fillna("") + " " +
            self.df["Answer"].fillna("")
        ).tolist()

        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print("Computing embeddings...")
        self.embeddings = self.embedder.encode(
            self.texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

        print("Initializing Groq client...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set")

        self.client = Groq(api_key=api_key)

    def retrieve(self, query: str, k: int = TOP_K) -> np.ndarray:
        """
        Retrieve indices of K most similar documents.

        Method:
          1. Encode query using same embedding model as corpus
          2. Compute L2 distance to all indexed documents
          3. Return indices of K nearest neighbors

        Parameters
        ----------
        query : str
            Natural language question from user.
        k : int
            Number of documents to retrieve (default: TOP_K).

        Returns
        -------
        np.ndarray
            Array of K document indices in rank order.
        """
        query_embedding = self.embedder.encode(
            [query], 
            convert_to_numpy=True
        )
        _, indices = self.index.search(query_embedding, k)
        return indices[0]

    def generate(self, query: str) -> tuple:
        """
        Generate answer using retrieved context.

        Pipeline:
          1. Retrieve K most relevant documents
          2. Extract Q&A pairs from knowledge base
          3. Format as numbered list in prompt
          4. Construct instruction prompt for LLM
          5. Send to Groq API with temperature=0.4 (deterministic)
          6. Return answer and associated sources

        Parameters
        ----------
        query : str
            Natural language question from user.

        Returns
        -------
        tuple
            (generated_answer, list_of_source_urls)

        Notes
        -----
        - The LLM is instructed to use only provided context
        - If answer is not in context, LLM states it explicitly
        - Temperature is set to 0.4 for consistent, factual responses
        """
        retrieved_ids = self.retrieve(query)

        context_blocks = []
        sources = set()

        for idx in retrieved_ids:
            row = self.df.iloc[idx]
            context_blocks.append(
                f"Question: {row['Question']}\nAnswer: {row['Answer']}"
            )

            if "Source_URLs" in row and pd.notna(row["Source_URLs"]):
                sources.add(row["Source_URLs"])

        context = "\n\n".join(context_blocks)

        prompt = f"""You are an educational assistant.
Answer the user's question using only the information provided below.
If the answer is not present in the context, state: "I do not have this information in my knowledge base."

Context:
{context}

Question:
{query}

Answer:"""

        completion = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )

        answer = completion.choices[0].message.content.strip()
        return answer, sorted(sources)


# =========================================================================
# INTERACTIVE APPLICATION
# =========================================================================

def main():
    """
    Main application loop for interactive chatbot.

    Workflow:
      1. Validate environment configuration
      2. Load knowledge base from CSV
      3. Initialize RAG chatbot
      4. Accept user queries in loop
      5. Retrieve context and generate answers
      6. Display answer and source attribution
      7. Exit on 'quit' or Ctrl+C

    Environment Variables
    --------------------
    GROQ_API_KEY : str
        Required. API key for Groq API access.

    Raises
    ------
    FileNotFoundError
        If CSV_FILE does not exist.
    RuntimeError
        If GROQ_API_KEY is not set.
    """

    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is not set.")
        print("Set it with: export GROQ_API_KEY='your_key'")
        sys.exit(1)

    print("=" * 70)
    print("INTERACTIVE RAG CHATBOT (Groq + FAISS)")
    print("=" * 70)
    print()

    try:
        df = pd.read_csv(CSV_FILE)
        print(f"Loaded dataset with {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: File not found: {CSV_FILE}")
        sys.exit(1)

    chatbot = RAGChatbot(df)

    print("\nChatbot ready. Type 'exit' or 'quit' to end session.")
    print("-" * 70)
    print()

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in {"exit", "quit", "q"}:
                print("\nSession ended.")
                break

            if not query:
                continue

            answer, sources = chatbot.generate(query)

            print("\nAnswer:")
            print(answer)

            if sources:
                print("\nSources:")
                for src in sources:
                    print(f"  - {src}")

            print()

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
