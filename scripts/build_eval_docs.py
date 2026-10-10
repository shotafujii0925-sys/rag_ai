"""Generate the published evaluation summary from its CSV, without API calls."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def render_summary(rows):
    selected = [(600, 100, 1), (600, 100, 4), (300, 100, 4), (900, 100, 4), (1200, 200, 8)]
    index = {(int(r["chunk_size"]), int(r["chunk_overlap"]), int(r["top_k"])): r for r in rows}
    lines = ["# RAG検索評価（2026-10-10再計算）", "",
             "チャンク欠落修正後の48条件 × 20問、ローカルBM25のみ。",
             "[CSV](rag-eval-2026-10-10.csv)と[実行情報・ケース別結果](rag-eval-2026-10-10.json)を同梱。",
             "この表は `python scripts/build_eval_docs.py` でCSVから生成します。", "",
             "| chunk / overlap / k | hit | MRR | context precision | context recall | p95 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in selected:
        r = index[key]
        lines.append(f"| {' / '.join(map(str, key))} | {float(r['hit_rate']):.2f} | "
                     f"{float(r['mrr']):.3f} | {float(r['context_precision']):.3f} | "
                     f"{float(r['context_recall']):.3f} | {float(r['latency_ms_p95']):.2f}ms |")
    lines += ["", "Context Precisionは取得チャンクのうち正解文書由来の割合による近似です。",
              "Context Recallは想定回答の各文のトークンが文脈に含まれるかによる近似です。",
              "kとチャンク分割を変えると両者も変わり、kを増やすと必ずprecisionが下がるわけではありません。",
              "FaithfulnessとAnswer Relevancyは未計測（空欄）。生成品質の改善はこの実験だけでは判断できません。",
              "レイテンシは実行環境に依存します。旧2026-09-07 CSVは修正前の歴史的結果として保持しています。", ""]
    return "\n".join(lines)


def main():
    with (ROOT / "docs/rag-eval-2026-10-10.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    (ROOT / "docs/rag-eval-summary.md").write_text(render_summary(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
