"""Indexing and retrieval.

BM25 over character bigrams is the default and needs no network. When cloud mode
is on, embeddings are blended in. Retrieval is the one part of the pipeline that
must work with no key at all, so the lexical path is never optional.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..config import settings
from .loader import Document
from .splitter import Chunk, split_documents, tokenize

BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(frozen=True)
class Passage:
    source: str
    text: str
    score: float
    excerpt: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "text": self.text,
            "score": round(self.score, 4),
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    passages: list[Passage]
    elapsed_ms: float
    method: str

    @property
    def is_empty(self) -> bool:
        return not self.passages


def excerpt_for(query: str, text: str, *, length: int = 220) -> str:
    if len(text) <= length:
        return text
    terms = [term for term in tokenize(query) if len(term) > 1]
    lowered = text.lower()
    position = next((lowered.find(term) for term in terms if term in lowered), -1)
    if position < 0:
        return f"{text[:length].strip()}…"
    start = max(0, position - length // 3)
    end = min(len(text), start + length)
    return f"{'…' if start else ''}{text[start:end].strip()}{'…' if end < len(text) else ''}"


class Retriever:
    """An in-memory index over the loaded documents."""

    def __init__(
        self,
        documents: list[Document],
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
        use_embeddings: bool | None = None,
    ) -> None:
        if not documents:
            raise ValueError("参照文書が登録されていません。")
        self.chunks: list[Chunk] = split_documents(
            documents, chunk_size=chunk_size, overlap=overlap
        )
        if not self.chunks:
            raise ValueError("参照文書から本文を取り出せませんでした。")
        self._tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self._lengths = [len(tokens) or 1 for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths)
        config = settings()
        self.use_embeddings = config.is_cloud if use_embeddings is None else use_embeddings
        self._vectors: list[list[float]] | None = None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({chunk.source for chunk in self.chunks})

    def _bm25(self, query: str) -> list[float]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return [0.0] * len(self.chunks)

        document_frequency = Counter(
            term for tokens in self._tokens for term in set(tokens) & query_terms
        )
        scores: list[float] = []
        for tokens, length in zip(self._tokens, self._lengths):
            counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(
                    1
                    + (len(self.chunks) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequency + BM25_K1 * (
                    1 - BM25_B + BM25_B * length / self._average_length
                )
                score += idf * frequency * (BM25_K1 + 1) / denominator
            scores.append(score)

        peak = max(scores)
        return [score / peak for score in scores] if peak > 0 else scores

    def _semantic(self, query: str) -> list[float] | None:
        if not self.use_embeddings:
            return None
        from ..llm import embed

        if self._vectors is None:
            self._vectors = embed([chunk.text for chunk in self.chunks])
        query_vector = embed([query])[0]
        return [_cosine(query_vector, vector) for vector in self._vectors]

    def retrieve(
        self, query: str, *, top_k: int | None = None, threshold: float | None = None
    ) -> RetrievalResult:
        """Return the best passages for a query.

        `threshold` is *relative*: scores are normalized by the best match in this
        search, so the top hit is always 1.0 and only lower-ranked passages can be
        cut. The one case where everything is filtered out is a query that shares
        no term with any chunk — then every score is 0 and the result is empty,
        which is what "no relevant document" means here.
        """
        config = settings()
        question = str(query or "").strip()
        if not question:
            raise ValueError("質問を入力してください。")

        limit = config.top_k if top_k is None else top_k
        floor = config.score_threshold if threshold is None else threshold

        started = time.perf_counter()
        lexical = self._bm25(question)
        semantic = self._semantic(question)
        if semantic:
            combined = [0.6 * s + 0.4 * lexical[i] for i, s in enumerate(semantic)]
            method = "hybrid"
        else:
            combined = lexical
            method = "bm25"

        ranked = sorted(range(len(self.chunks)), key=lambda i: combined[i], reverse=True)
        passages = [
            Passage(
                source=self.chunks[index].source,
                text=self.chunks[index].text,
                score=combined[index],
                excerpt=excerpt_for(question, self.chunks[index].text),
            )
            for index in ranked[:limit]
            if combined[index] > floor
        ]
        return RetrievalResult(
            query=question,
            passages=passages,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            method=method,
        )


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = (sum(a * a for a in left) ** 0.5) * (sum(b * b for b in right) ** 0.5)
    return max(0.0, min(1.0, dot / norm)) if norm else 0.0
