import streamlit as st

from views._shared import load_resources, setup
from src.config import settings

setup(
    "はじめに",
    "ミナモ サポート研修AI",
    "架空のクラウド予約管理サービス「ミナモ」のカスタマーサポート担当者向け研修デモです。"
    "AIが顧客役を務め、公開用のFAQと対応マニュアルをRAGで参照しながら模擬対応を行い、"
    "終了後に5つの観点で自動評価します。",
)

retriever, scenarios, criteria = load_resources()
config = settings()

st.markdown("## 研修の流れ")
steps = [
    ("1. シナリオを選ぶ", "利用方法・契約変更・不具合・苦情の4区分から選びます。難易度が異なります。"),
    ("2. 顧客役と対話する", "AIが演じる顧客に、担当者として応対します。文字入力のほか、クラウドモードでは音声入力も使えます。"),
    ("3. 対話を終了する", "十分に応対できたと思ったら終了します。途中で終えても評価できます。"),
    ("4. 評価を確認する", "5観点の点数と、点数の理由・改善アドバイスが表示されます。"),
]
for title, body in steps:
    st.markdown(f'<div class="step"><b>{title}</b><span>{body}</span></div>', unsafe_allow_html=True)

st.markdown("## 主な機能")
left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        '<div class="card"><b>RAGによる根拠付き応対</b><br>'
        "公開FAQと対応マニュアルを検索し、根拠のある応対を練習します。"
        "取得された文書と類似度は技術デモ画面で確認できます。</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card"><b>5観点の自動評価</b><br>'
        "説明の正確性・顧客への配慮・課題把握・問題解決力・根拠に基づく回答を採点します。</div>",
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        '<div class="card"><b>ローカルモード</b><br>'
        "APIキーがなくても、検索・対話・評価の一連の流れを確認できます。"
        "顧客の応答はシナリオに用意した台本を再生します。</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card"><b>RAG性能の定量評価</b><br>'
        "チャンクサイズや取得件数を変えて、検索品質と応答時間を比較できます。</div>",
        unsafe_allow_html=True,
    )

st.markdown("## 収録シナリオ")
st.dataframe(
    [
        {
            "区分": scenario.category,
            "タイトル": scenario.title,
            "難易度": scenario.difficulty,
            "顧客像": scenario.persona.name,
        }
        for scenario in scenarios
    ],
    hide_index=True,
    use_container_width=True,
)

st.markdown("## 注意事項")
st.info(
    "**このアプリで扱うサービス・FAQ・マニュアル・顧客・シナリオはすべて架空です。**\n\n"
    "実在する企業、製品、人物とは関係ありません。研修用のデモであり、"
    "実際の顧客対応の判断に使用しないでください。"
)
st.caption(
    f"現在のモード: {'クラウド（AI生成あり）' if config.is_cloud else 'ローカル（外部API呼び出しなし）'} / "
    f"参照文書: {'、'.join(retriever.sources)} / 索引チャンク数: {retriever.chunk_count}"
)

if st.button("模擬対応を始める", type="primary"):
    st.switch_page("views/training.py")
