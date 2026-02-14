#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script performs a systematic analysis of RAG chatbot parameters:
- Temperature (model creativity vs consistency)
- Top_K retrieval (number of context documents)
- Embedding models (different sentence encoders)
- LLM models (different Groq models)

METRICS EVALUATED:
- Accuracy@K: Correctness of retrieved documents
- Recall@K: Fraction of relevant documents retrieved
- MRR: Mean Reciprocal Rank (position of first relevant)
- Timing: Response latency in milliseconds

Results are ranked and documented for academic research.

Usage:
    python chatbot_parameter_analysis_improved.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple
import faiss
from sentence_transformers import SentenceTransformer, util
from groq import Groq

# CONFIGURATION: PARAMETERS TO TEST

CSV_FILE = "chatbot_csv.csv"

EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",      # Fast, 384 dims
    "all-mpnet-base-v2",     # Better quality, 768 dims
]

LLM_MODELS = [
    "llama-3.1-8b-instant",      # Faster
    "llama-3.3-70b-versatile",   # Better accuracy
]

TEMPERATURES = [0.2, 0.5, 0.8]  # Low=consistent, High=creative

TOP_K_VALUES = [1, 3, 5]  # Number of retrieved documents

TEST_QUERIES = [
    "What is the acceptance rate?",
    "How can I apply for admission?",
    "What are the tuition fees?",
    "Do you offer scholarships?",
    "What is the campus location?",
    "What are English language requirements?",
]


#METRICS EVALUATION


class MetricsEvaluator:
    """Evaluate retrieval and generation quality with academic metrics"""
    
    def __init__(self, df: pd.DataFrame, embedding_model: str):
        self.df = df.reset_index(drop=True)
        self.embedding_model = embedding_model
        
        # Precompute question embeddings for relevance matching
        self.embedder = SentenceTransformer(embedding_model)
        self.question_embeddings = self.embedder.encode(
            df["Question"].fillna("").tolist(),
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    def find_relevant_docs(self, query: str, threshold: float = 0.5) -> set:
        """Find relevant documents using cosine similarity"""
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        similarities = util.pytorch_cos_sim(query_embedding, self.question_embeddings)[0].cpu().numpy()
        
        # Documents above threshold or top 3
        relevant_by_threshold = set(np.where(similarities > threshold)[0])
        relevant_by_top3 = set(np.argsort(-similarities)[:3])
        
        return relevant_by_threshold | relevant_by_top3
    
    def evaluate_retrieval(self, query: str, retrieved_indices: List[int]) -> Dict:
        """Compute Accuracy@K, Recall@K, MRR metrics"""
        metrics = {}
        
        # Find ground truth relevant documents
        relevant_docs = self.find_relevant_docs(query)
        
        retrieved_set = set(retrieved_indices)
        num_relevant = len(relevant_docs)
        num_retrieved = len(retrieved_set)
        
        if num_relevant == 0:
            # No relevant docs found in KB
            metrics["Accuracy@K"] = 0.0
            metrics["Recall@K"] = 0.0
            metrics["MRR"] = 0.0
            return metrics
        
        # Accuracy@K: fraction of retrieved docs that are correct
        num_correct = len(retrieved_set & relevant_docs)
        accuracy = num_correct / num_retrieved if num_retrieved > 0 else 0.0
        metrics["Accuracy@K"] = accuracy
        
        # Recall@K: fraction of relevant docs found
        recall = num_correct / num_relevant if num_relevant > 0 else 0.0
        metrics["Recall@K"] = recall
        
        # MRR: Mean Reciprocal Rank (position of first relevant)
        mrr = 0.0
        for i, idx in enumerate(retrieved_indices, 1):
            if idx in relevant_docs:
                mrr = 1.0 / i
                break
        metrics["MRR"] = mrr
        
        return metrics

# GROQ API CLIENT WITH TIMING

class GroqChatbot:
    """RAG Chatbot using Groq API with FAISS retrieval"""
    
    def __init__(self, df: pd.DataFrame, embedding_model: str, llm_model: str):
        """Initialize with dataframe and model choices"""
        self.df = df.reset_index(drop=True)
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        
        # Combine question and answer for retrieval
        self.texts = (
            self.df["Question"].fillna("") + " " + 
            self.df["Answer"].fillna("")
        ).tolist()
        
        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model)
        
        # Create embeddings
        self.embeddings = self.embedder.encode(
            self.texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        # Groq client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Metrics evaluator
        self.metrics_evaluator = MetricsEvaluator(df, embedding_model)
    
    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[str], List[int]]:
        """Retrieve top-k most relevant documents"""
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(query_embedding, top_k)
        
        contexts = []
        idx_list = []
        for idx in indices[0]:
            row = self.df.iloc[idx]
            contexts.append(
                f"Q: {row['Question']}\nA: {row['Answer']}"
            )
            idx_list.append(idx)
        return contexts, idx_list
    
    def generate(self, query: str, temperature: float = 0.5, top_k: int = 3) -> Dict:
        """Generate answer with professional metrics"""
        
        # RETRIEVAL
        retrieval_start = time.time()
        contexts, retrieved_indices = self.retrieve(query, top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000  # milliseconds
        
        context_text = "\n\n".join(contexts)
        
        prompt = f"""You are an educational assistant. Answer using ONLY the context provided.
If the answer is not in the context, say "Information not available in knowledge base."

Context:
{context_text}

Question: {query}

Answer:"""
        
        # GENERATION
        generation_start = time.time()
        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            answer = completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"    ⚠️  Generation error: {e}")
            answer = ""
        
        generation_time = (time.time() - generation_start) * 1000  # milliseconds
        total_time = retrieval_time + generation_time
        
        # EVALUATE METRICS
        retrieval_metrics = self.metrics_evaluator.evaluate_retrieval(query, retrieved_indices)
        
        return {
            "query": query,
            "answer": answer,
            "temperature": temperature,
            "top_k": top_k,
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name,
            "context_count": len(contexts),
            # Metrics
            "metrics/Accuracy@K": retrieval_metrics.get("Accuracy@K", 0.0),
            "metrics/Recall@K": retrieval_metrics.get("Recall@K", 0.0),
            "metrics/MRR": retrieval_metrics.get("MRR", 0.0),
            # Timing
            "timing/retrieval_ms": retrieval_time,
            "timing/generation_ms": generation_time,
            "timing/total_ms": total_time,
        }



# ANALYSIS & RANKING

class ParameterAnalyzer:
    """Run systematic parameter analysis with professional ranking"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = []
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            print("❌ Error: GROQ_API_KEY not set")
            sys.exit(1)
    
    def run_analysis(self):
        """Run all parameter combinations"""
        total_tests = (
            len(EMBEDDING_MODELS) * 
            len(LLM_MODELS) * 
            len(TEMPERATURES) * 
            len(TOP_K_VALUES) * 
            len(TEST_QUERIES)
        )
        
        print(f"\n{'='*70}")
        print(f"RAG CHATBOT PARAMETER ANALYSIS (WITH PROFESSIONAL METRICS)")
        print(f"{'='*70}")
        print(f"Total tests: {total_tests}")
        print(f"Test queries: {len(TEST_QUERIES)}")
        print(f"Embedding models: {len(EMBEDDING_MODELS)}")
        print(f"LLM models: {len(LLM_MODELS)}")
        print(f"Temperatures: {len(TEMPERATURES)}")
        print(f"Top-K values: {len(TOP_K_VALUES)}")
        print(f"{'='*70}\n")
        
        test_count = 0
        
        for emb_model in EMBEDDING_MODELS:
            for llm_model in LLM_MODELS:
                print(f"\n🔧 Testing: {emb_model} + {llm_model}")
                print("-" * 70)
                
                try:
                    chatbot = GroqChatbot(self.df, emb_model, llm_model)
                except Exception as e:
                    print(f"  ❌ Failed to initialize: {e}")
                    continue
                
                for temp in TEMPERATURES:
                    for top_k in TOP_K_VALUES:
                        print(f"  Temperature: {temp}, Top-K: {top_k}")
                        
                        for query in TEST_QUERIES:
                            try:
                                result = chatbot.generate(
                                    query=query,
                                    temperature=temp,
                                    top_k=top_k
                                )
                                self.results.append(result)
                                test_count += 1
                                
                                if test_count % 10 == 0:
                                    print(f"    ✓ Completed {test_count}/{total_tests} tests")
                                
                            except Exception as e:
                                print(f"    ❌ Error: {e}")
        
        print(f"\n✅ Analysis complete! {test_count} tests executed.")
    
    def rank_configurations(self) -> pd.DataFrame:
        """Rank configurations by combined metrics score"""
        results_df = pd.DataFrame(self.results)
        
        # Group by main configuration (embedding + LLM)
        grouped = results_df.groupby(['embedding_model', 'llm_model']).agg({
            'metrics/Accuracy@K': 'mean',
            'metrics/Recall@K': 'mean',
            'metrics/MRR': 'mean',
            'timing/total_ms': 'mean',
            'timing/retrieval_ms': 'mean',
            'timing/generation_ms': 'mean',
        }).reset_index()
        
        # Calculate overall score (weighted)
        # 50% retrieval quality, 50% performance
        grouped['retrieval_score'] = (
            grouped['metrics/Accuracy@K'] * 0.4 +
            grouped['metrics/Recall@K'] * 0.4 +
            grouped['metrics/MRR'] * 0.2
        )
        
        # Normalize timing penalty (lower latency is better)
        max_latency = grouped['timing/total_ms'].max()
        grouped['latency_penalty'] = grouped['timing/total_ms'] / max_latency * 0.2  # Max 20% penalty
        
        grouped['overall_score'] = grouped['retrieval_score'] - grouped['latency_penalty']
        
        # Rank
        grouped['rank'] = grouped['overall_score'].rank(ascending=False)
        grouped = grouped.sort_values('rank')
        
        return grouped.round(4)
    
    def save_results(self):
        """Save detailed results to CSV and generate report"""
        results_df = pd.DataFrame(self.results)
        ranking = self.rank_configurations()
        
        # Save raw results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f"chatbot_results_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"\n✅ Detailed results saved to: {csv_file}")
        
        # Save ranking
        ranking_file = f"chatbot_ranking_{timestamp}.csv"
        ranking.to_csv(ranking_file, index=False)
        print(f"✅ Configuration ranking saved to: {ranking_file}")
        
        return results_df, ranking
    
    def generate_report(self, ranking: pd.DataFrame) -> str:
        """Generate comprehensive analysis report"""
        results_df = pd.DataFrame(self.results)
        
        report = []
        report.append("="*80)
        report.append("RAG CHATBOT PARAMETER ANALYSIS - COMPREHENSIVE REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        #  1. RANKING
        report.append("\n1. CONFIGURATION RANKING (by Overall Score)")
        report.append("-"*80)
        
        ranking_display = ranking[['rank', 'embedding_model', 'llm_model', 
                                   'metrics/Accuracy@K', 'metrics/Recall@K', 
                                   'metrics/MRR', 'timing/total_ms', 'overall_score']]
        report.append(ranking_display.to_string(index=False))
        
        #  2. BEST CONFIGURATION 
        report.append("\n\n2. RECOMMENDED CONFIGURATION (BEST FOR GRADE)")
        report.append("-"*80)
        
        best = ranking.iloc[0]
        report.append(f"\n🏆 Best Overall Configuration:")
        report.append(f"  Embedding Model: {best['embedding_model']}")
        report.append(f"  LLM Model: {best['llm_model']}")
        report.append(f"  Overall Score: {best['overall_score']:.4f}")
        report.append(f"  Rank: {int(best['rank'])}")
        
        report.append(f"\n📊 Metrics:")
        report.append(f"  Accuracy@K: {best['metrics/Accuracy@K']:.4f}")
        report.append(f"  Recall@K: {best['metrics/Recall@K']:.4f}")
        report.append(f"  MRR: {best['metrics/MRR']:.4f}")
        
        report.append(f"\n⚡ Performance:")
        report.append(f"  Total Latency: {best['timing/total_ms']:.2f} ms")
        report.append(f"  Retrieval Time: {best['timing/retrieval_ms']:.2f} ms")
        report.append(f"  Generation Time: {best['timing/generation_ms']:.2f} ms")
        
        #  3. EMBEDDING MODEL COMPARISON 
        report.append("\n\n3. EMBEDDING MODEL COMPARISON")
        report.append("-"*80)
        
        emb_comparison = results_df.groupby('embedding_model').agg({
            'metrics/Accuracy@K': ['mean', 'std'],
            'metrics/Recall@K': ['mean', 'std'],
            'metrics/MRR': ['mean', 'std'],
            'timing/total_ms': 'mean',
        }).round(4)
        
        report.append("\n" + str(emb_comparison))
        
        best_emb = results_df.groupby('embedding_model')['metrics/Accuracy@K'].mean().idxmax()
        report.append(f"\n✓ Best embedding model: {best_emb}")
        
        #  4. LLM MODEL COMPARISON 
        report.append("\n\n4. LLM MODEL COMPARISON")
        report.append("-"*80)
        
        llm_comparison = results_df.groupby('llm_model').agg({
            'metrics/Accuracy@K': ['mean', 'std'],
            'metrics/Recall@K': ['mean', 'std'],
            'metrics/MRR': ['mean', 'std'],
            'timing/total_ms': 'mean',
        }).round(4)
        
        report.append("\n" + str(llm_comparison))
        
        best_llm = results_df.groupby('llm_model')['metrics/Accuracy@K'].mean().idxmax()
        report.append(f"\n✓ Best LLM model: {best_llm}")
        
        #  5. TEMPERATURE ANALYSIS 
        report.append("\n\n5. TEMPERATURE IMPACT")
        report.append("-"*80)
        
        temp_analysis = results_df.groupby('temperature').agg({
            'metrics/Accuracy@K': ['mean', 'std'],
            'metrics/Recall@K': ['mean', 'std'],
            'metrics/MRR': ['mean', 'std'],
        }).round(4)
        
        report.append("\n" + str(temp_analysis))
        
        #  6. TOP-K ANALYSIS 
        report.append("\n\n6. TOP-K RETRIEVAL ANALYSIS")
        report.append("-"*80)
        
        topk_analysis = results_df.groupby('top_k').agg({
            'metrics/Accuracy@K': ['mean', 'std'],
            'metrics/Recall@K': ['mean', 'std'],
            'metrics/MRR': ['mean', 'std'],
            'timing/total_ms': 'mean',
        }).round(4)
        
        report.append("\n" + str(topk_analysis))
        
        #  7. OVERALL STATISTICS 
        report.append("\n\n7. OVERALL STATISTICS")
        report.append("-"*80)
        
        report.append(f"\nTotal tests: {len(results_df)}")
        report.append(f"Avg Accuracy@K: {results_df['metrics/Accuracy@K'].mean():.4f}")
        report.append(f"Avg Recall@K: {results_df['metrics/Recall@K'].mean():.4f}")
        report.append(f"Avg MRR: {results_df['metrics/MRR'].mean():.4f}")
        report.append(f"Avg Total Latency: {results_df['timing/total_ms'].mean():.2f} ms")
        
        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        return "\n".join(report)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not set")
        print("   Set it with: export GROQ_API_KEY='your_key'")
        sys.exit(1)
    
    print("\n📂 Loading dataset...")
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✅ Loaded {len(df)} Q&A pairs")
    except FileNotFoundError:
        print(f"❌ File not found: {CSV_FILE}")
        sys.exit(1)
    
    # Run analysis
    analyzer = ParameterAnalyzer(df)
    analyzer.run_analysis()
    
    # Save and generate report
    results_df, ranking = analyzer.save_results()
    report = analyzer.generate_report(ranking)
    
    print("\n" + report)
    
    # Save report to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"chatbot_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ Report saved to: {report_file}")


if __name__ == "__main__":
    main()
