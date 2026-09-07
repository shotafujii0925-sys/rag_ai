"""Run the labelled RAG evaluation set and save the result.

    python scripts/run_rag_eval.py                       # 検索指標のみ（APIキー不要）
    python scripts/run_rag_eval.py --chunk-sizes 400,600,900 --top-k 3,4,8
    python scripts/run_rag_eval.py --with-generation     # 生成指標も測る（APIキーが必要・課金あり）

Results land in experiments/<UTC timestamp>/ as results.json and results.csv.
Metrics that were not measured are written as empty, never as a placeholder value.
"""

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, settings  # noqa: E402
from src.rag.evaluator import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from src.rag.generator import answer_question  # noqa: E402
from src.rag.loader import load_json, load_knowledge  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402

COLUMNS = [
    "chunk_size", "chunk_overlap", "top_k", "method", "chunks", "cases",
    "hit_rate", "passage_hit_rate", "mrr",
    "context_precision", "context_recall",
    "faithfulness", "answer_relevancy",
    "latency_ms_mean", "latency_ms_p95", "errors",
]


def numbers(raw: str, cast):
    values = [cast(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"値が空です: {raw!r}")
    return values


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def run_condition(cases, documents, chunk_size, overlap, top_k, with_generation):
    retriever = Retriever(documents, chunk_size=chunk_size, overlap=overlap)
    rows, latencies, errors = [], [], []
    faith_scores, relevancy_scores = [], []

    for case in cases:
        started = time.perf_counter()
        result = retriever.retrieve(case["question"], top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)

        sources = [passage.source for passage in result.passages]
        rank = next((i for i, s in enumerate(sources, 1) if s == case["source"]), None)
        precision = context_precision(result.passages, case["source"])
        recall = context_recall(case["ground_truth"], [p.text for p in result.passages])

        row = {
            "id": case["id"],
            "retrieved": sources,
            "rank": rank,
            "passage_hit": any(case["must_contain"] in p.text for p in result.passages),
            "context_precision": precision["score"],
            "context_recall": recall["score"],
            "latency_ms": latencies[-1],
            "faithfulness": None,
            "answer_relevancy": None,
        }

        if with_generation:
            try:
                answer = answer_question(case["question"], retriever, top_k=top_k)
                if answer.passages:
                    contexts = [passage.text for passage in answer.passages]
                    row["faithfulness"] = faithfulness(answer.text, contexts)["score"]
                    row["answer_relevancy"] = answer_relevancy(case["question"], answer.text)["score"]
                    faith_scores.append(row["faithfulness"])
                    relevancy_scores.append(row["answer_relevancy"])
            except Exception as error:  # 1件の失敗で全体を止めない
                errors.append({"case_id": case["id"], "error": type(error).__name__})

        rows.append(row)

    hits = [row for row in rows if row["rank"]]
    return rows, {
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "top_k": top_k,
        "method": "hybrid" if retriever.use_embeddings else "bm25",
        "chunks": retriever.chunk_count,
        "cases": len(rows),
        "hit_rate": len(hits) / len(rows),
        "passage_hit_rate": sum(row["passage_hit"] for row in rows) / len(rows),
        "mrr": sum(1 / row["rank"] for row in hits) / len(rows),
        "context_precision": sum(row["context_precision"] for row in rows) / len(rows),
        "context_recall": sum(row["context_recall"] for row in rows) / len(rows),
        "faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else None,
        "answer_relevancy": (
            sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None
        ),
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p95": percentile(latencies, 0.95),
        "errors": len(errors),
    }, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config = settings()
    parser.add_argument("--chunk-sizes", default=str(config.chunk_size))
    parser.add_argument("--overlaps", default=str(config.chunk_overlap))
    parser.add_argument("--top-k", default=str(config.top_k))
    parser.add_argument("--cases", type=Path, default=config.data_dir / "eval" / "rag_eval_set.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Faithfulness と Answer Relevancy も測る（APIキーが必要）",
    )
    args = parser.parse_args()

    if args.with_generation and not config.is_cloud:
        parser.error(
            "--with-generation には APP_MODE=cloud と OPENAI_API_KEY が必要です。"
        )

    cases = load_json(args.cases)["cases"]
    documents = load_knowledge()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, detail = [], []
    for chunk_size in numbers(args.chunk_sizes, int):
        for overlap in numbers(args.overlaps, int):
            if overlap >= chunk_size:
                print(f"skip: overlap {overlap} >= chunk_size {chunk_size}")
                continue
            for top_k in numbers(args.top_k, int):
                rows, summary, errors = run_condition(
                    cases, documents, chunk_size, overlap, top_k, args.with_generation
                )
                summaries.append(summary)
                detail.append({"summary": summary, "cases": rows, "errors": errors})
                print(
                    f"size={chunk_size} overlap={overlap} k={top_k}: "
                    f"hit={summary['hit_rate']:.2f} mrr={summary['mrr']:.3f} "
                    f"ctxP={summary['context_precision']:.3f} "
                    f"ctxR={summary['context_recall']:.3f} "
                    f"p95={summary['latency_ms_p95']:.0f}ms"
                )

    payload = {
        "run_at": stamp,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mode": config.mode,
            "chat_model": config.chat_model if args.with_generation else None,
            "embedding_model": config.embedding_model if config.is_cloud else None,
        },
        "eval_set": args.cases.name,
        "case_count": len(cases),
        "generation_metrics_measured": bool(args.with_generation),
        "conditions": detail,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\n{len(summaries)} 条件を {out_dir} に保存しました。")
    if not args.with_generation:
        print("Faithfulness / Answer Relevancy は未計測（空欄）です。")


if __name__ == "__main__":
    main()
