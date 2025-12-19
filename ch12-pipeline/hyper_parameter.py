import os
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List

import numpy as np
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import BatchEvalRunner, SemanticSimilarityEvaluator
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI


@dataclass
class RunResult:
    score: float
    params: Dict[str, Any]


class ParamTuner:
    def __init__(
        self,
        param_fn: Callable[[Dict[str, Any]], RunResult],
        param_dict: Dict[str, List[Any]],
        fixed_param_dict: Dict[str, Any],
        show_progress: bool = True,
    ):
        self.param_fn = param_fn
        self.param_dict = param_dict
        self.fixed_param_dict = fixed_param_dict
        self.show_progress = show_progress
        self._validate_params()

    def _validate_params(self) -> None:
        if not isinstance(self.param_dict, dict):
            raise ValueError("param_dict must be a dictionary")
        if not isinstance(self.fixed_param_dict, dict):
            raise ValueError("fixed_param_dict must be a dictionary")

    def tune(self) -> List[RunResult]:
        param_names = list(self.param_dict.keys())
        param_values = list(self.param_dict.values())
        results: List[RunResult] = []

        for combination in product(*param_values):
            params_dict = dict(zip(param_names, combination))
            full_params = {**params_dict, **self.fixed_param_dict}

            if self.show_progress:
                print(f"Evaluating parameters: {params_dict}")

            result = self.param_fn(full_params)
            results.append(result)

        return results


def _build_index(chunk_size: int, documents: List[Any]) -> VectorStoreIndex:
    """Build a vector store index from documents with given chunk size."""
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
    doc_objs = [doc if isinstance(doc, Document) else Document(text=str(doc)) for doc in documents]
    nodes = node_parser.get_nodes_from_documents(doc_objs)
    return VectorStoreIndex(nodes)


def get_responses(questions: List[str], query_engine, show_progress: bool = False) -> List[Any]:
    """Get responses for a list of questions using the query engine."""
    responses = []
    for question in questions:
        if show_progress:
            print(f"- Querying: {question}")
        responses.append(query_engine.query(question))
    return responses


def _get_eval_batch_runner_semantic_similarity() -> BatchEvalRunner:
    evaluator = SemanticSimilarityEvaluator()
    return BatchEvalRunner({"semantic_similarity": evaluator}, workers=2, show_progress=True)


def objective_function_semantic_similarity(params_dict: Dict[str, Any]) -> RunResult:
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    index = _build_index(chunk_size, docs)
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    pred_response_objs = get_responses(eval_qs, query_engine, show_progress=True)

    eval_batch_runner = _get_eval_batch_runner_semantic_similarity()
    eval_results = eval_batch_runner.evaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )

    mean_score = np.array([r.score for r in eval_results["semantic_similarity"]]).mean()
    return RunResult(score=mean_score, params=params_dict)


def main() -> None:
    # Sample documents and evaluation data for demonstration only
    # in practice, use your own documents and evaluation questions/responses

    documents = [
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris.",
        "France is known for its wine and cheese.",
        "The Louvre Museum houses the Mona Lisa.",
        "French is the official language of France.",
    ]

    eval_qs = [
        "What is the capital of France?",
        "Where is the Eiffel Tower located?",
        "What is France known for?",
    ]

    ref_response_strs = [
        "The capital of France is Paris.",
        "The Eiffel Tower is located in Paris.",
        "France is known for its wine and cheese.",
    ]

    param_dict = {"chunk_size": [256, 512, 1024], "top_k": [1, 2, 5]}
    fixed_param_dict = {
        "docs": documents,
        "eval_qs": eval_qs,
        "ref_response_strs": ref_response_strs,
    }

    param_tuner = ParamTuner(
        param_fn=objective_function_semantic_similarity,
        param_dict=param_dict,
        fixed_param_dict=fixed_param_dict,
        show_progress=True,
    )

    results = param_tuner.tune()

    print("\nTuning Results:")
    for result in results:
        print(f"Parameters: {result.params}, Score: {result.score:.4f}")

    best_result = max(results, key=lambda x: x.score)
    print(f"\nBest Parameters: {best_result.params}, Best Score: {best_result.score:.4f}")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Use a concrete LLM so downstream components have one
    Settings.llm = OpenAI(model="gpt-4o-mini")

    main()
