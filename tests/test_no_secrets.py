"""公開リポジトリとしての健全性。

秘密情報、個人情報、そして前身プロジェクト（金融IT企業のインターンで開発した非公開
アプリ）に由来する語彙が、追跡対象のファイルに入っていないことを検査する。手作業の
監査は一度きりだが、テストなら毎回走る。
"""

import re
import subprocess

import pytest

from src.config import REPO_ROOT

SECRETS = {
    "OpenAI key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "private key block": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "bearer token": re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
    "assigned secret": re.compile(
        r"""(?i)(api[_-]?key|password|token|secret|connection_?string)\s*[:=]\s*["'][^"'\s]{12,}["']"""
    ),
}

# 前身プロジェクト由来の固有名詞・業務語彙。
LEGACY = {
    "employer resource": re.compile(r"(?i)aiintern|asrs-|nft-ai|teamB"),
    "org unit": re.compile(r"AI推進室|2026インターン|チームB"),
    "brand": re.compile(r"つくる∞|make, create, innovate"),
    "internal deployment": re.compile(r"gpt-5\.4|gpt-realtime-2\.1"),
    "azure endpoint": re.compile(
        r"(?i)search\.windows\.net|openai\.azure\.com|blob\.core\.windows\.net"
    ),
    "azure setting": re.compile(r"AZURE_[A-Z_]+"),
    "loan domain": re.compile(r"住宅ローン|カードローン|事業性融資|団体信用生命保険|行員|融資"),
    "original check id": re.compile(r"EXP-\d{2,}"),
}

CLOUD_IDENTIFIERS = re.compile(r"(?i)tenant[_-]?id|subscription[_-]?id|resource[_-]?group")

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"0\d{1,4}-\d{1,4}-\d{4}")

ALLOWED_EMAIL_DOMAINS = ("example.com", "example.co.jp", "example.jp", "users.noreply.github.com")
BINARY_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".woff", ".woff2")


def tracked_files() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if output.returncode != 0:
        pytest.skip("git リポジトリではないため、追跡ファイルの検査をスキップします。")
    return [path for path in output.stdout.split("\0") if path]


@pytest.fixture(scope="module")
def documents() -> list[tuple[str, str]]:
    files = tracked_files()
    if not files:
        pytest.skip("追跡ファイルがありません。")
    collected = []
    for path in files:
        if path.endswith(BINARY_SUFFIXES):
            continue
        full = REPO_ROOT / path
        if not full.exists():
            continue
        try:
            collected.append((path, full.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    return collected


@pytest.mark.parametrize("label", sorted(SECRETS))
def test_no_credentials_in_tracked_files(documents, label):
    pattern = SECRETS[label]
    hits = [path for path, body in documents if pattern.search(body)]
    assert not hits, f"{label} らしき文字列が {hits} にあります。"


@pytest.mark.parametrize("label", sorted(LEGACY))
def test_no_predecessor_project_traces(documents, label):
    pattern = LEGACY[label]
    hits = [path for path, body in documents if pattern.search(body)]
    # このテスト自身は検出パターンを持つので除外する。
    hits = [path for path in hits if not path.endswith("test_no_secrets.py")]
    assert not hits, f"前身プロジェクト由来の記述（{label}）が {hits} にあります。"


def test_no_cloud_account_identifiers(documents):
    hits = [
        path
        for path, body in documents
        if CLOUD_IDENTIFIERS.search(body) and not path.endswith("test_no_secrets.py")
    ]
    assert not hits, f"クラウドのアカウント識別子らしき記述が {hits} にあります。"


def test_no_real_email_addresses(documents):
    found = {
        (path, match)
        for path, body in documents
        for match in EMAIL.findall(body)
        if not match.endswith(ALLOWED_EMAIL_DOMAINS) and not path.endswith("test_no_secrets.py")
    }
    assert not found, f"メールアドレスが含まれています: {sorted(found)}"


def test_no_phone_numbers(documents):
    hits = [
        path
        for path, body in documents
        if PHONE.search(body) and not path.endswith("test_no_secrets.py")
    ]
    assert not hits, f"電話番号らしき記述が {hits} にあります。"


def test_env_file_is_not_tracked():
    assert ".env" not in tracked_files()


def test_streamlit_secrets_is_not_tracked():
    assert ".streamlit/secrets.toml" not in tracked_files()


def test_env_example_has_no_real_key():
    body = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in body
    assert not SECRETS["OpenAI key"].search(body)


def test_gitignore_covers_the_secret_locations():
    body = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for required in (".env", ".streamlit/secrets.toml", "experiments/", "data/uploads/"):
        assert required in body, f".gitignore に {required} がありません。"


def test_public_data_is_marked_as_fictional():
    for name in ("public_faq.md", "support_manual.md"):
        body = (REPO_ROOT / "data" / name).read_text(encoding="utf-8")
        assert "架空" in body, f"{name} に架空である旨の記載がありません。"
