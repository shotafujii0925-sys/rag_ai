"""Document loading: read bytes or files into normalized plain text."""

import html as html_module
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from ..config import settings

SUPPORTED_SUFFIXES = (".md", ".txt", ".html", ".htm")

WHITESPACE = re.compile(r"[ \t  -​　]+")
BLANK_LINES = re.compile(r"\n{3,}")
_CHARSET = re.compile(rb"""charset\s*=\s*["']?\s*([\w-]+)""", re.IGNORECASE)

_SKIP_TAGS = frozenset(
    {"head", "title", "script", "style", "noscript", "template", "svg",
     "nav", "header", "footer", "aside", "form"}
)
_BLOCK_TAGS = frozenset(
    {"p", "div", "section", "article", "main", "blockquote", "pre",
     "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "dl", "dt", "dd",
     "table", "thead", "tbody", "tr", "hr", "br"}
)


@dataclass(frozen=True)
class Document:
    name: str
    text: str


def normalize(text: str) -> str:
    collapsed = WHITESPACE.sub(" ", str(text or "").replace("\r\n", "\n").replace("\r", "\n"))
    return BLANK_LINES.sub("\n\n", collapsed).strip()


class _TextExtractor(HTMLParser):
    """Pull readable text out of HTML, dropping scripts, chrome, and markup.

    Also the sanitization step for anything we might render: tags never survive,
    so an uploaded document cannot inject markup into the page.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in {"td", "th"} and self.parts:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    return normalize("".join(parser.parts))


def decode(data: bytes) -> str:
    match = _CHARSET.search(data[:4096])
    if match:
        try:
            return data.decode(match.group(1).decode("ascii"), errors="replace")
        except LookupError:
            pass
    return data.decode("utf-8", errors="replace")


def read_bytes(name: str, data: bytes) -> Document:
    """Turn an uploaded file into a Document, enforcing type and size limits."""
    clean_name = Path(str(name or "")).name.strip() or "untitled"
    if not clean_name.lower().endswith(SUPPORTED_SUFFIXES):
        allowed = " / ".join(SUPPORTED_SUFFIXES)
        raise ValueError(f"対応していない形式です。{allowed} のいずれかを選んでください。")

    limit = settings().max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise ValueError(f"ファイルが大きすぎます。{settings().max_upload_mb}MB 以下にしてください。")

    body = decode(data)
    text = html_to_text(body) if clean_name.lower().endswith((".html", ".htm")) else normalize(body)
    if not text:
        raise ValueError(f"「{clean_name}」に本文がありません。")
    return Document(name=clean_name, text=text)


def read_file(path: Path) -> Document:
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path.name}")
    return read_bytes(path.name, path.read_bytes())


def load_knowledge(directory: Path | None = None) -> list[Document]:
    """Load the bundled public knowledge base used by local mode."""
    base = directory or settings().data_dir
    documents = [read_file(path) for path in sorted(base.glob("*.md"))]
    if not documents:
        raise FileNotFoundError("参照文書が見つかりません。data ディレクトリを確認してください。")
    return documents


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name} をJSONとして読めませんでした。") from error


def escape_for_display(text: str) -> str:
    """Escape text that will be placed inside HTML we generate."""
    return html_module.escape(str(text or ""), quote=True)
