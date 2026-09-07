"""Document splitting and the tokenizer the lexical index is built on."""

import re
from dataclasses import dataclass

from ..config import settings
from .loader import Document, normalize

CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿]+")
LATIN_TOKEN = re.compile(r"[a-z0-9][a-z0-9'_-]*")


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str
    position: int


def tokenize(text: str) -> list[str]:
    """Split into match units, using character bigrams for CJK runs.

    Japanese has no word delimiters. Bigrams give the lexical scorer something to
    match on without shipping a morphological analyzer and its dictionary.
    """
    lowered = str(text or "").lower()
    tokens = LATIN_TOKEN.findall(lowered)
    for run in CJK_RUN.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def split_text(text: str, *, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split into overlapping windows, preferring heading and sentence boundaries."""
    config = settings()
    size = config.chunk_size if chunk_size is None else chunk_size
    step_back = config.chunk_overlap if overlap is None else overlap

    if size <= 0:
        raise ValueError("chunk_size は 1 以上にしてください。")
    if not 0 <= step_back < size:
        raise ValueError("overlap は 0 以上 chunk_size 未満にしてください。")

    body = normalize(text)
    if not body:
        return []
    if len(body) <= size:
        return [body]

    step = size - step_back
    chunks: list[str] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + size)
        if end < len(body):
            window = body[start:end]
            boundary = max(window.rfind("\n#"), window.rfind("\n\n"), window.rfind("。"))
            if boundary > size // 2:
                end = start + boundary + 1
        piece = body[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(body):
            break
        start = max(start + step, end - step_back)
    return chunks


def split_documents(
    documents: list[Document], *, chunk_size: int | None = None, overlap: int | None = None
) -> list[Chunk]:
    return [
        Chunk(source=document.name, text=piece, position=index)
        for document in documents
        for index, piece in enumerate(
            split_text(document.text, chunk_size=chunk_size, overlap=overlap)
        )
    ]
