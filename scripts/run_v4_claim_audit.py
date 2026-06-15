from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "OneDrive" / "Desktop"
PAPER_DIR = REPO_ROOT / "paper" / "iclr2027"
REPO_PDF = REPO_ROOT / "paper" / "final" / "best-of-n-llm-v4.pdf"
DESKTOP_PDF = DESKTOP / "best-of-n-llm-v4.pdf"
SOURCE_MAP = DESKTOP / "PAPER_SOURCE_MAP.md"
SUMMARY = REPO_ROOT / "results" / "v4_protocol_evidence" / "summary.json"

STALE_PATTERNS = (
    "best-of-n-llm-" + "v" + "2.pdf",
    "best-of-n-llm-" + "v" + "2-source.zip",
    "best-of-n-llm-" + "v" + "3.pdf",
    "best-of-n-llm-" + "v" + "3-source.zip",
    "18_" + "v" + "3_cached_evidence.py",
    "results/" + "v" + "3_cached_evidence",
    "V" + "Three",
    "Inference Value " + "Theorem",
    "inference value " + "theorem",
    "iclr" + "_submission",
)

SCAN_ROOTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs",
    REPO_ROOT / "paper" / "iclr2027",
    REPO_ROOT / "scripts",
    REPO_ROOT / "experiments" / "18_v4_protocol_evidence.py",
)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_count(path: Path) -> int:
    try:
        completed = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raw = path.read_bytes()
        return len(re.findall(rb"/Type\s*/Page\b", raw))
    match = re.search(r"^Pages:\s+(\d+)$", completed.stdout, re.MULTILINE)
    if not match:
        fail(f"could not parse page count from pdfinfo for {path}")
    return int(match.group(1))


def iter_scan_files(root: Path):
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".tex", ".json", ".ps1", ".py"}:
            yield path


def audit_pdfs() -> None:
    if not REPO_PDF.exists():
        fail(f"missing repo PDF: {REPO_PDF}")
    if not DESKTOP_PDF.exists():
        fail(f"missing Desktop PDF: {DESKTOP_PDF}")
    repo_pages = page_count(REPO_PDF)
    desktop_pages = page_count(DESKTOP_PDF)
    if repo_pages < 25:
        fail(f"repo PDF has {repo_pages} pages, expected at least 25")
    if desktop_pages != repo_pages:
        fail(f"Desktop page count {desktop_pages} differs from repo page count {repo_pages}")
    if sha256(REPO_PDF) != sha256(DESKTOP_PDF):
        fail("repo PDF and Desktop PDF hashes differ")


def audit_summary() -> None:
    if not SUMMARY.exists():
        fail(f"missing v4 summary: {SUMMARY}")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    core = summary.get("core", {})
    heldout = summary.get("heldout_4096", {})
    gates = summary.get("gates", {})
    cross = summary.get("cross_benchmark", {})
    cards = summary.get("benchmark_cards", {})
    live = summary.get("live_judge", {})
    if core.get("n_triples") != 35964:
        fail("unexpected exact-law triple count")
    if float(core.get("exact_law_mae", 1.0)) > 0.001:
        fail("exact-law MAE exceeds v4 tolerance")
    if float(core.get("auc_only_mae_N48", 0.0)) < 0.05:
        fail("AUC failure control is too small; high-N distinction may be broken")
    if heldout.get("records") != 119:
        fail("held-out 4096-depth slice record count is not 119")
    if int(gates.get("claim_pass", -1)) != 8 or int(gates.get("claim_missing", -1)) != 4:
        fail("claim-gate pass/missing counts changed unexpectedly")
    if int(cross.get("families", 0)) < 4 or int(cross.get("records", 0)) < 1280:
        fail("cross-benchmark pilot coverage is below v4 expectation")
    if int(cards.get("families", 0)) < 5 or int(cards.get("records", 0)) < 1399:
        fail("benchmark-card ledger is missing MATH plus real benchmark families")
    if float(cards.get("min_coverage", 0.0)) < 1.0:
        fail("benchmark-card ledger includes incomplete benchmark coverage")
    if int(live.get("pairs", 0)) != 135 or int(live.get("judgments", 0)) != 6480:
        fail("live-judge scoped subset changed unexpectedly")


def audit_stale_text() -> None:
    hits: list[str] = []
    for root in SCAN_ROOTS:
        for path in iter_scan_files(root):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in STALE_PATTERNS:
                if pattern in text:
                    rel = path.relative_to(REPO_ROOT)
                    hits.append(f"{rel}: {pattern}")
    if hits:
        fail("stale v2/title strings remain:\n" + "\n".join(hits))


def audit_source_map() -> None:
    if not SOURCE_MAP.exists():
        fail(f"missing Desktop source map: {SOURCE_MAP}")
    text = SOURCE_MAP.read_text(encoding="utf-8")
    expected_parts = (
        "best-of-n-llm-v4.pdf",
        "C:\\Users\\wangz\\Downloads\\best-of-n-llm",
        "Jason-Wang313/best-of-n-llm",
    )
    if not all(part in text for part in expected_parts):
        fail("Desktop source map does not point best-of-n-llm-v4.pdf to the local folder and GitHub repo")


def audit_latex_log() -> None:
    log_path = PAPER_DIR / "main.log"
    if not log_path.exists():
        fail(f"missing LaTeX log: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    bad = re.findall(
        r"(undefined|Citation|Overfull|Fatal|Emergency|LaTeX Warning|Package natbib Warning|Package hyperref Warning)",
        text,
        flags=re.IGNORECASE,
    )
    if bad:
        fail(f"LaTeX log contains blocking warnings: {sorted(set(bad))}")


def main() -> None:
    audit_pdfs()
    audit_summary()
    audit_stale_text()
    audit_source_map()
    audit_latex_log()
    print("submission audit complete: best-of-n-llm v4")


if __name__ == "__main__":
    main()
