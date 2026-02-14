
# ============================================================================
# tfidf_chatbot.py - Reusable TF-IDF Chatbot Module
# ============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple

class FAQChatbot:
    """
    FAQ Retrieval-based Chatbot using TF-IDF similarity matching.

    This chatbot matches user queries against a FAQ database using
    TF-IDF vectorization and cosine similarity to find the best answer.

    Attributes:
        vectorizer: Fitted TF-IDF vectorizer
        question_vectors: TF-IDF vectors for all FAQ questions
        questions: Array of FAQ questions
        answers: Array of FAQ answers
        categories: Array of category labels
        source_urls: Array of source URLs
        languages: Array of language codes
        query_count: Number of queries processed
    """

    def __init__(self, vectorizer, question_vectors, questions, answers, 
                 categories, source_urls, languages):
        """Initialize the chatbot with trained vectorizer and FAQ data."""
        self.vectorizer = vectorizer
        self.question_vectors = question_vectors
        self.questions = questions
        self.answers = answers
        self.categories = categories
        self.source_urls = source_urls
        self.languages = languages
        self.query_count = 0

    def find_best_answer(self, user_query: str, top_k: int = 1) -> List[Dict]:
        """
        Find the best matching answer(s) for a user query.

        Args:
            user_query (str): The user's question
            top_k (int): Number of top matches to return (default=1)

        Returns:
            List[Dict]: List of results with answer, score, category, and source
        """
        self.query_count += 1

        # Vectorize user query
        query_vector = self.vectorizer.transform([user_query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            result = {
                'rank': rank,
                'similarity_score': float(similarities[idx]),
                'matched_question': self.questions[idx],
                'answer': self.answers[idx],
                'category': self.categories[idx],
                'language': 'English' if self.languages[idx] == 'en' else 'Polish',
                'source_url': self.source_urls[idx],
                'confidence': self._get_confidence(similarities[idx])
            }
            results.append(result)

        return results

    def _get_confidence(self, similarity_score: float) -> str:
        """Determine confidence level based on similarity score."""
        if similarity_score > 0.3:
            return 'High'
        elif similarity_score > 0.15:
            return 'Medium'
        else:
            return 'Low'

    def get_stats(self) -> Dict:
        """Return chatbot statistics."""
        return {
            'total_queries': self.query_count,
            'faq_size': len(self.questions),
            'vocabulary_size': len(self.vectorizer.get_feature_names_out()),
            'model_type': 'TF-IDF Cosine Similarity'
        }

    def batch_query(self, queries: List[str]) -> List[List[Dict]]:
        """Process multiple queries at once."""
        return [self.find_best_answer(q, top_k=1) for q in queries]
