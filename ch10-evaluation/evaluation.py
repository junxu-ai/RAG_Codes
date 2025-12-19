# Step 1: Install Necessary Libraries
#
# pip install ragas ragchecker trulens datasets langchain-openai openai scikit-learn

"""
Step 2: Import Required Libraries and Configure the Environment

Import the necessary libraries and set up your environment variables, such as the OpenAI API key.
"""
import os
from typing import Dict, List

import numpy as np
from datasets import Dataset, load_dataset
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
)
from ragchecker import RAGChecker
from sklearn.model_selection import train_test_split


def _ensure_api_key() -> None:
    """Ensure the OpenAI key is present; keep setup out of code for safety."""
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")


def _build_records(raw_ds, llm_raw: ChatOpenAI, max_samples: int = 30) -> List[Dict]:
    """
    Step 3: Create Benchmark Dataset for Evaluation

    Convert a HF dataset into the columns expected by ragas metrics.
    """
    records: List[Dict] = []
    for i, row in enumerate(raw_ds):
        if i >= max_samples:
            break

        question = row.get("question") or ""
        # Amnesty QA has "answer" and "context"; fall back to empty defaults if missing.
        reference = row.get("answer") or row.get("answers") or ""
        contexts = row.get("context") or row.get("contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        elif not isinstance(contexts, list):
            contexts = [str(contexts)]

        prompt = (
            "Use the provided contexts to answer the question concisely.\n\n"
            f"Question: {question}\n\n"
            "Contexts:\n" + "\n\n".join(f"- {c}" for c in contexts) + "\n\nAnswer:"
        )
        response = llm_raw.invoke([HumanMessage(content=prompt)]).content.strip()

        records.append(
            {
                "user_input": question,
                "response": response,
                "retrieved_contexts": contexts,
                # Helpful extras for analysis
                "reference": reference,
                "reference_contexts": contexts,
            }
        )
    return records


def _split_dataset(records: List[Dict]) -> (Dataset, Dataset):
    """
    Step 4: Split Benchmark Dataset into Evaluation and Test Sets
    """
    eval_records, test_records = train_test_split(records, test_size=0.2, random_state=42)
    return Dataset.from_list(eval_records), Dataset.from_list(test_records)


def main() -> None:
    _ensure_api_key()

    # Initialize the Language Model
    llm_raw = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm = LangchainLLMWrapper(llm_raw)

    """
    Step 3 (continued): Load benchmark data and build evaluation dataset
    """
    raw_ds = load_dataset("explodinggradients/amnesty_qa", "english_v3", split="eval")
    records = _build_records(raw_ds, llm_raw, max_samples=30)
    eval_dataset, test_dataset = _split_dataset(records)

    """
    Step 5–7: Generate responses (above), parse components (implicit), and evaluate accuracy
    """
    metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm)]
    results = evaluate(eval_dataset, metrics)
    for metric in metrics:
        scores = results[metric.name]
        print(f"{metric.name} Average Score: {np.mean(scores):.4f}")

    """
    Step 8: Evaluate Semantic/Context Precision, Recall, and Accuracy
    """
    context_metrics = [ContextPrecision(llm=llm), ContextRecall(llm=llm), NoiseSensitivity(llm=llm)]
    context_results = evaluate(eval_dataset, context_metrics)
    for metric in context_metrics:
        scores = context_results[metric.name]
        print(f"{metric.name} Average Score: {np.mean(scores):.4f}")

    """
    Step 9: Analyze Failed Cases Using Diagnostic Metrics
    """
    rag_checker = RAGChecker(llm=llm)
    diagnostics = rag_checker.analyze_failures(eval_dataset, results)
    for issue in diagnostics:
        print(issue)

    """
    Step 10–11: Refine and report on the held-out test set
    """
    test_results = evaluate(test_dataset, metrics + context_metrics)
    for metric in metrics + context_metrics:
        scores = test_results[metric.name]
        print(f"{metric.name} Test Average Score: {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
