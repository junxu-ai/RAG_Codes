"""
Minimal RAG evaluation example with Ragas.

Prereqs:
  pip install ragas ragchecker trulens datasets langchain-openai openai scikit-learn
  export OPENAI_API_KEY=...

This script:
- Builds a tiny in-memory corpus and a trivial retriever
- Generates answers with ChatOpenAI
- Evaluates with Ragas metrics (faithfulness, answer relevancy, context precision/recall, noise sensitivity)
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from datasets import Dataset
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    NoiseSensitivity,
)
from ragchecker import RAGChecker
from sklearn.model_selection import train_test_split


@dataclass
class CorpusDoc:
    id: str
    text: str


class SimpleRetriever:
    """Naive retriever: returns top-k docs by length (placeholder for a real vector store)."""

    def __init__(self, docs: List[CorpusDoc], top_k: int = 2):
        self.docs = docs
        self.top_k = top_k

    def retrieve(self, question: str) -> List[str]:
        # Just return the longest docs; replace with your vector search
        ranked = sorted(self.docs, key=lambda d: len(d.text), reverse=True)
        return [d.text for d in ranked[: self.top_k]]


def generate_answer(llm_raw: ChatOpenAI, question: str, contexts: List[str]) -> str:
    prompt = (
        "Use the provided contexts to answer the question concisely.\n\n"
        f"Question: {question}\n\n"
        "Contexts:\n" + "\n\n".join(f"- {c}" for c in contexts) + "\n\nAnswer:"
    )
    return llm_raw.invoke([HumanMessage(content=prompt)]).content.strip()


def build_eval_dataset(
    llm_raw: ChatOpenAI, retriever: SimpleRetriever, qa_pairs: List[Dict[str, str]]
) -> Dataset:
    """Build a HuggingFace Dataset with the columns expected by ragas metrics."""
    records = []
    for pair in qa_pairs:
        user_input = pair["question"]
        ground_truth = pair["ground_truth"]
        retrieved_contexts = retriever.retrieve(user_input)
        response = generate_answer(llm_raw, user_input, retrieved_contexts)

        records.append(
            {
                # Required by faithfulness/answer_relevancy in newer ragas
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                # Helpful additional fields
                "reference": ground_truth,
                "reference_contexts": retrieved_contexts,
            }
        )

    return Dataset.from_list(records)


def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Tiny demo corpus
    corpus = [
        CorpusDoc("1", "The capital of France is Paris."),
        CorpusDoc("2", "The Eiffel Tower is located in Paris, France."),
        CorpusDoc("3", "France is renowned for its wine, cheese, and rich culture."),
        CorpusDoc("4", "The Louvre Museum in Paris houses the Mona Lisa."),
    ]

    qa_pairs = [
        {"question": "What is the capital of France?", "ground_truth": "Paris."},
        {"question": "Where is the Eiffel Tower located?", "ground_truth": "Paris, France."},
        {
            "question": "What is France known for?",
            "ground_truth": "Wine, cheese, and rich culture.",
        },
    ]

    llm_raw = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    llm = LangchainLLMWrapper(llm_raw)
    retriever = SimpleRetriever(corpus, top_k=2)

    # Split into eval/test
    eval_pairs, test_pairs = train_test_split(qa_pairs, test_size=0.33, random_state=42)

    eval_dataset = build_eval_dataset(llm_raw, retriever, eval_pairs)
    test_dataset = build_eval_dataset(llm_raw, retriever, test_pairs)

    base_metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm)]
    ctx_metrics = [ContextPrecision(llm=llm), ContextRecall(llm=llm), NoiseSensitivity(llm=llm)]

    print("=== Eval set ===")
    eval_results = evaluate(eval_dataset, base_metrics + ctx_metrics)
    for metric in base_metrics + ctx_metrics:
        scores = eval_results[metric.name]
        print(f"{metric.name} Average Score: {np.mean(scores):.4f}")

    print("\n=== Test set ===")
    test_results = evaluate(test_dataset, base_metrics + ctx_metrics)
    for metric in base_metrics + ctx_metrics:
        scores = test_results[metric.name]
        print(f"{metric.name} Test Average Score: {np.mean(scores):.4f}")

    # Optional diagnostics
    rag_checker = RAGChecker(llm=llm)
    diagnostics = rag_checker.analyze_failures(eval_dataset, eval_results)
    for issue in diagnostics:
        print(issue)


if __name__ == "__main__":
    main()
