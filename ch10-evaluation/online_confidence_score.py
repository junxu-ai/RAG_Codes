import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass
class RAGConfig:
    w_retrieval: float = 0.25
    w_coverage: float = 0.15
    w_groundedness: float = 0.30
    w_relevance: float = 0.20
    w_uncertainty: float = 0.10
    top_k: int = 5
    use_judge_fallback: bool = True
    judge_model: str = "gpt-4.1-mini"


class RAGConfidenceScorer:
    def __init__(
        self,
        config: RAGConfig = RAGConfig(),
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.cfg = config
        self.embedder = SentenceTransformer(embed_model) if SentenceTransformer else None
        self.nli = None
        if pipeline:
            try:
                self.nli = pipeline(
                    "text-classification",
                    model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1",
                    return_all_scores=True,
                )
            except Exception:
                self.nli = None
        self.client = OpenAI() if (OpenAI and os.getenv("OPENAI_API_KEY")) else None

    def _norm01(self, x, lo=None, hi=None):
        x = np.array(x, dtype=float)
        if lo is None:
            lo = float(np.min(x))
        if hi is None:
            hi = float(np.max(x))
        if hi == lo:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo)

    def _embed(self, texts: List[str]):
        if not self.embedder:
            return None
        return self.embedder.encode(texts, normalize_embeddings=True)

    def retrieval_relevance(self, query: str, retrieved_chunks: List[Dict[str, Any]]):
        scores = np.array([float(c.get("score", 0.0)) for c in retrieved_chunks], dtype=float)
        if len(scores) == 0:
            return 0.0, {"reason": "no_retrieval_scores"}

        top1 = float(scores[0])
        gap = float(scores[0] - scores[1]) if len(scores) > 1 else float(scores[0])
        dense_sparse_agreement = float(
            np.mean([
                1.0 if c.get("dense_hit", True) and c.get("sparse_hit", True) else 0.0
                for c in retrieved_chunks[:self.cfg.top_k]
            ])
        )

        vals = self._norm01([top1, gap, dense_sparse_agreement])
        R = float(0.4 * vals[0] + 0.3 * vals[1] + 0.3 * vals[2])
        return R, {"top1": top1, "gap": gap, "agreement": dense_sparse_agreement}

    def coverage(self, query: str, retrieved_chunks: List[Dict[str, Any]], expected_entities: Optional[List[str]] = None):
        context = " ".join([c.get("text", "") for c in retrieved_chunks])
        q_terms = [t.lower() for t in re.findall(r"\b\w+\b", query) if len(t) > 3]

        if expected_entities is None:
            expected_entities = list(dict.fromkeys(re.findall(r"\b[A-Z][A-Za-z0-9_\-]+\b", query)))

        entity_hits = sum(1 for e in expected_entities if e.lower() in context.lower())
        entity_recall = entity_hits / max(len(expected_entities), 1)

        term_hits = sum(1 for t in q_terms if t in context.lower())
        term_recall = term_hits / max(len(q_terms), 1)

        C = float(0.5 * entity_recall + 0.5 * term_recall)
        return C, {"entity_recall": entity_recall, "term_recall": term_recall}

    def _sentence_split(self, text: str):
        s = re.split(r"(?<=[.!?])\s+", text.strip())
        return [x.strip() for x in s if x.strip()]

    def groundedness(self, answer: str, retrieved_chunks: List[Dict[str, Any]]):
        context = " ".join([c.get("text", "") for c in retrieved_chunks])
        claims = self._sentence_split(answer)
        if not claims:
            return 0.0, {"claims": 0}

        if self.nli:
            scores = []
            for claim in claims:
                premise = context[:4000]
                out = self.nli({"text": claim, "text_pair": premise})
                label_scores = out[0]
                label_scores = {d["label"].lower(): d["score"] for d in label_scores}
                entail = max(label_scores.get("entailment", 0.0), label_scores.get("supported", 0.0))
                contra = label_scores.get("contradiction", 0.0)
                scores.append(max(0.0, entail - contra))
            G = float(np.mean(scores))
            return G, {"claims": len(claims), "mode": "nli"}

        if self.embedder:
            ce = self._embed(claims)
            te = self._embed([context])[0:1]
            sim = cosine_similarity(ce, te).ravel()
            G = float(np.mean(np.clip(sim, 0, 1)))
            return G, {"claims": len(claims), "mode": "embedding_proxy"}

        if self.cfg.use_judge_fallback and self.client:
            prompt = (
                f"Rate groundedness from 0 to 1.\n"
                f"Answer: {answer}\nContext: {context}\n"
                f"Return JSON {{'score': number, 'reason': string}}"
            )
            resp = self.client.chat.completions.create(model=self.cfg.judge_model, messages=[{"role": "user", "content": prompt}])
            text = resp.choices[0].message.content
            try:
                data = json.loads(text)
                return float(data["score"]), {"mode": "judge", "reason": data.get("reason", "")}
            except Exception:
                m = re.search(r"([01](?:\.\d+)?)", text)
                return float(m.group(1)) if m else 0.0, {"mode": "judge_raw"}

        return 0.0, {"mode": "fallback_zero"}

    def answer_relevance(self, query: str, answer: str):
        if self.embedder:
            q = self._embed([query])[0:1]
            a = self._embed([answer])[0:1]
            sim = float(cosine_similarity(q, a)[0, 0])
            return float(np.clip((sim + 1) / 2, 0, 1)), {"mode": "embedding"}

        if self.cfg.use_judge_fallback and self.client:
            prompt = (
                f"Rate answer relevance from 0 to 1.\n"
                f"Query: {query}\nAnswer: {answer}\n"
                f"Return JSON {{'score': number}}"
            )
            resp = self.client.chat.completions.create(model=self.cfg.judge_model, messages=[{"role": "user", "content": prompt}])
            text = resp.choices[0].message.content
            try:
                return float(json.loads(text)["score"]), {"mode": "judge"}
            except Exception:
                m = re.search(r"([01](?:\.\d+)?)", text)
                return float(m.group(1)) if m else 0.0, {"mode": "judge_raw"}

        return 0.0, {"mode": "fallback_zero"}

    def uncertainty(self, answer: str, logprobs: Optional[List[float]] = None, sample_answers: Optional[List[str]] = None):
        parts = []

        if logprobs:
            probs = np.exp(np.array(logprobs, dtype=float))
            token_conf = float(np.clip(np.mean(probs), 0, 1))
            parts.append(token_conf)

        if sample_answers and len(sample_answers) > 1 and self.embedder:
            emb = self._embed(sample_answers)
            sims = cosine_similarity(emb, emb)
            tri = sims[np.triu_indices_from(sims, k=1)]
            self_cons = float(np.clip(np.mean(tri), 0, 1)) if len(tri) else 0.0
            parts.append(self_cons)

        if parts:
            return float(np.mean(parts)), {"mode": "statistical", "parts": parts}

        if self.cfg.use_judge_fallback and self.client:
            prompt = (
                f"Estimate answer certainty from 0 to 1 where 1 means highly certain and grounded.\n"
                f"Answer: {answer}\n"
                f"Return JSON {{'score': number}}"
            )
            resp = self.client.chat.completions.create(model=self.cfg.judge_model, messages=[{"role": "user", "content": prompt}])
            text = resp.choices[0].message.content
            try:
                return float(json.loads(text)["score"]), {"mode": "judge"}
            except Exception:
                m = re.search(r"([01](?:\.\d+)?)", text)
                return float(m.group(1)) if m else 0.0, {"mode": "judge_raw"}

        return 0.0, {"mode": "fallback_zero"}

    def score(self, query: str, answer: str, retrieved_chunks: List[Dict[str, Any]], expected_entities=None, logprobs=None, sample_answers=None):
        R, rd = self.retrieval_relevance(query, retrieved_chunks)
        C, cd = self.coverage(query, retrieved_chunks, expected_entities)
        G, gd = self.groundedness(answer, retrieved_chunks)
        A, ad = self.answer_relevance(query, answer)
        U, ud = self.uncertainty(answer, logprobs, sample_answers)

        final = float(
            self.cfg.w_retrieval * R +
            self.cfg.w_coverage * C +
            self.cfg.w_groundedness * G +
            self.cfg.w_relevance * A +
            self.cfg.w_uncertainty * U
        )

        band = "high" if final >= 0.85 else "medium" if final >= 0.65 else "low" if final >= 0.40 else "abstain"

        return {
            "final_confidence": final,
            "band": band,
            "metrics": {
                "retrieval_relevance": R,
                "coverage": C,
                "groundedness": G,
                "answer_relevance": A,
                "uncertainty": U,
            },
            "details": {
                "retrieval": rd,
                "coverage": cd,
                "groundedness": gd,
                "answer_relevance": ad,
                "uncertainty": ud,
            },
        }


# Example
query = "What is the refund policy for annual plans?"
answer = "Annual plans can be refunded within 14 days of purchase if no more than 20% of usage has occurred."
retrieved_chunks = [
    {"text": "Annual subscriptions are eligible for a refund within 14 days.", "score": 0.91, "dense_hit": True, "sparse_hit": True},
    {"text": "Usage above 20% may make refunds ineligible.", "score": 0.84, "dense_hit": True, "sparse_hit": True},
    {"text": "Monthly plans are handled separately.", "score": 0.32, "dense_hit": False, "sparse_hit": True},
]

scorer = RAGConfidenceScorer()
result = scorer.score(query, answer, retrieved_chunks, expected_entities=["refund", "annual", "plans"])
print(result)