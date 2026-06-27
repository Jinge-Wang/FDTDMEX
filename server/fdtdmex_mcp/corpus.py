"""Corpus loader + BM25 search for the FDTDMEX MCP discovery server.

`search_docs` / `get_doc` are backed by a documentation/example corpus **generated from
real on-disk sources** — never retyped:

- `examples/<dir>/README.md` + the example script (the verified O-band ring study and the
  other `examples/` entries) → refs ``example/<dir>`` and ``example/<dir>.py``,
- `docs/*.md` guides → refs ``doc/<stem>``,
- the `fdtdmex.io` `SceneModel` schema + the `JsonSetup` accepted-type whitelist
  (`src/fdtdx/conversion/json.py`) → ref ``schema/scene_model``,
- key fdtdx/fdtdmex module + class docstrings → refs ``api/<name>``.

Ranking is a hand-written BM25 index (mirrors ``mesa-mcp/mesa_mcp/docs/index.py``); the
expensive walk is cached to a user cache dir keyed by a cheap content signature
(file count / total size / max mtime) and the index is memoized per process and built
lazily on first search. This module NEVER imports ag-fdtd.

Regenerate explicitly with ``python scripts/build_corpus.py`` (or it rebuilds itself the
first time a source file changes).
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from pathlib import Path

# Bump when the corpus *format* changes so stale on-disk caches are rejected.
_CORPUS_VERSION = "1"

_BM25_K1 = 1.5
_BM25_B = 0.75
_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Per-process memo: repo-root -> (signature, Corpus)
_MEMO: dict = {}


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
def repo_root() -> Path:
    """Locate the FDTDMEX repo root (env override, else walk up for examples+docs)."""
    env = os.environ.get("FDTDMEX_REPO_ROOT")
    if env and Path(env).is_dir():
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "examples").is_dir() and (parent / "docs").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    d = Path(base) / "fdtdmex_mcp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(root: Path) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(root))
    return cache_dir() / f"corpus-{_CORPUS_VERSION}-{safe}.json"


# --------------------------------------------------------------------------- #
# Tokenization + snippets (mirrors the mesa-mcp pattern)
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> list[str]:
    """Lowercase [a-z0-9_] tokens; underscore compounds also yield their parts."""
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        if "_" in tok:
            tokens.extend(p for p in tok.split("_") if p)
    return tokens


def _snippet(text: str, q_terms: list[str], width: int = 220) -> str:
    """A cleaned one-line snippet centered on the first matching query term."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    low = cleaned.lower()
    pos = -1
    for t in q_terms:
        pos = low.find(t)
        if pos != -1:
            break
    if pos == -1:
        return cleaned[:width] + ("…" if len(cleaned) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(cleaned), start + width)
    return ("…" if start else "") + cleaned[start:end] + ("…" if end < len(cleaned) else "")


# --------------------------------------------------------------------------- #
# Markdown / docstring helpers
# --------------------------------------------------------------------------- #
def _md_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip() or fallback
        if s:
            return s[:120]
    return fallback


def _md_summary(text: str, fallback: str = "") -> str:
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("```"):
            continue
        return s[:200]
    return fallback


def _module_docstring(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return ""
    return ast.get_docstring(tree) or ""


def _class_docstring(path: Path, class_name: str) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_docstring(node) or ""
    return ""


def _class_fields(path: Path, class_name: str) -> list[str]:
    """Annotated field names of a class (e.g. the pydantic SceneModel), via AST."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    out.append(stmt.target.id)
            return out
    return []


def _named_sets(path: Path, names: tuple[str, ...]) -> dict[str, list[str]]:
    """String members of set-literal assignments to the given names (e.g. the
    `valid_object_names` / `valid_constraint_names` whitelist inside JsonSetup.validate)."""
    out: dict[str, list[str]] = {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Set):
            members = [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names and members:
                    out[tgt.id] = members
    return out


# --------------------------------------------------------------------------- #
# Source discovery — every entry is built from a real file on disk
# --------------------------------------------------------------------------- #
def _example_entries(root: Path) -> list[dict]:
    entries: list[dict] = []
    ex = root / "examples"
    if not ex.is_dir():
        return entries
    for d in sorted(p for p in ex.iterdir() if p.is_dir()):
        readme = d / "README.md"
        if readme.is_file():
            text = readme.read_text(encoding="utf-8", errors="replace")
            entries.append(
                {
                    "ref": f"example/{d.name}",
                    "section": "examples",
                    "title": _md_title(text, d.name),
                    "summary": _md_summary(text, f"{d.name} example."),
                    "source": str(readme),
                    "body": text,
                }
            )
        # primary script: <dirname>.py preferred, else the largest .py
        py = d / f"{d.name}.py"
        scripts = sorted(d.glob("*.py"), key=lambda p: p.stat().st_size, reverse=True)
        if not py.is_file() and scripts:
            py = scripts[0]
        if py.is_file():
            text = py.read_text(encoding="utf-8", errors="replace")
            doc = _module_docstring(py)
            entries.append(
                {
                    "ref": f"example/{d.name}.py",
                    "section": "examples",
                    "title": f"{d.name} — example script ({py.name})",
                    "summary": (doc.splitlines()[0][:200] if doc else f"Runnable {d.name} script."),
                    "source": str(py),
                    "body": text,
                }
            )
    return entries


def _doc_entries(root: Path) -> list[dict]:
    entries: list[dict] = []
    docs = root / "docs"
    if not docs.is_dir():
        return entries
    for md in sorted(docs.glob("*.md")):
        text = md.read_text(encoding="utf-8", errors="replace")
        entries.append(
            {
                "ref": f"doc/{md.stem}",
                "section": "guides",
                "title": _md_title(text, md.stem),
                "summary": _md_summary(text, f"{md.stem} guide."),
                "source": str(md),
                "body": text,
            }
        )
    return entries


def _schema_entry(root: Path) -> list[dict]:
    schema_py = root / "src" / "fdtdmex" / "io" / "schema.py"
    pack_py = root / "src" / "fdtdmex" / "io" / "pack.py"
    json_py = root / "src" / "fdtdx" / "conversion" / "json.py"
    if not schema_py.is_file():
        return []
    parts: list[str] = []
    mod_doc = _module_docstring(schema_py)
    if mod_doc:
        parts += [mod_doc, ""]
    scene_doc = _class_docstring(schema_py, "SceneModel")
    fields = _class_fields(schema_py, "SceneModel")
    parts.append("SceneModel — the resolved scene payload (fdtdmex.io):")
    if scene_doc:
        parts.append(scene_doc)
    if fields:
        parts.append("fields: " + ", ".join(fields))
    parts.append("")
    if pack_py.is_file():
        pack_doc = _module_docstring(pack_py)
        if pack_doc:
            parts += ["Config HDF5 layout (pack / sim_init output):", pack_doc, ""]
    if json_py.is_file():
        sets = _named_sets(json_py, ("valid_object_names", "valid_constraint_names"))
        if sets.get("valid_object_names"):
            parts.append("Accepted object types (JsonSetup.validate whitelist):")
            parts += [f"  {n}" for n in sets["valid_object_names"]]
            parts.append("")
        if sets.get("valid_constraint_names"):
            parts.append("Accepted constraint types:")
            parts += [f"  {n}" for n in sets["valid_constraint_names"]]
    body = "\n".join(parts).strip()
    if not body:
        return []
    return [
        {
            "ref": "schema/scene_model",
            "section": "schema",
            "title": "fdtdmex.io scene schema — SceneModel + accepted object/constraint types",
            "summary": "The resolved-scene payload (SceneModel) and the JsonSetup accepted-type whitelist.",
            "source": None,
            "body": body,
        }
    ]


# (ref, title-noun, module path under src/, optional class name)
_API_SPECS = [
    ("api/materials", "Material (isotropic / anisotropic / lossy)", "fdtdx/materials.py", "Material"),
    ("api/dispersion", "Drude-Lorentz ADE dispersion", "fdtdx/dispersion.py", None),
    (
        "api/phasor_detector",
        "PhasorDetector (complex DFT monitor)",
        "fdtdx/objects/detectors/phasor.py",
        "PhasorDetector",
    ),
]


def _api_entries(root: Path) -> list[dict]:
    entries: list[dict] = []
    for ref, noun, rel, cls in _API_SPECS:
        path = root / "src" / rel
        if not path.is_file():
            continue
        mod_doc = _module_docstring(path)
        cls_doc = _class_docstring(path, cls) if cls else ""
        body = "\n\n".join(p for p in (mod_doc, cls_doc) if p).strip()
        if not body:
            continue
        entries.append(
            {
                "ref": ref,
                "section": "api",
                "title": noun,
                "summary": (body.splitlines()[0][:200]),
                "source": None,
                "body": body,
            }
        )
    return entries


def _all_entries(root: Path) -> list[dict]:
    return _example_entries(root) + _doc_entries(root) + _schema_entry(root) + _api_entries(root)


# --------------------------------------------------------------------------- #
# Signature + cache
# --------------------------------------------------------------------------- #
def _signature(root: Path) -> dict:
    """Cheap fingerprint over the corpus source files to detect staleness."""
    count = total = 0
    max_mtime = 0.0
    roots = [root / "examples", root / "docs", root / "src" / "fdtdmex" / "io", root / "src" / "fdtdx" / "conversion"]
    roots += [root / "src" / Path(rel).parent for _, _, rel, _ in _API_SPECS]
    for base in roots:
        if not base.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith((".md", ".py")):
                    continue
                try:
                    st = os.stat(os.path.join(dirpath, fn))
                except OSError:
                    continue
                count += 1
                total += st.st_size
                max_mtime = max(max_mtime, st.st_mtime)
    return {"version": _CORPUS_VERSION, "count": count, "total": total, "max_mtime": round(max_mtime, 3)}


def _load_entries(root: Path, sig: dict, force: bool = False) -> list[dict]:
    cache = _cache_path(root)
    if not force:
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("signature") == sig:
                return cached["entries"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    entries = _all_entries(root)
    try:
        cache.write_text(json.dumps({"signature": sig, "entries": entries}), encoding="utf-8")
    except OSError:
        pass
    return entries


# --------------------------------------------------------------------------- #
# BM25 over one chunk per ref
# --------------------------------------------------------------------------- #
class Corpus:
    """In-memory BM25 index over the generated corpus (one chunk per ref)."""

    def __init__(self, entries: list[dict]):
        self.entries = entries
        self._by_ref = {e["ref"]: e for e in entries}
        self._titles = [e["title"].lower() for e in entries]
        self._postings: list[dict] = []
        self._lengths: list[int] = []
        self._df: dict[str, int] = {}
        for e in entries:
            tf: dict[str, int] = {}
            blob = f"{e['ref']} {e['title']} {e['summary']} {e['body']}"
            for t in _tokenize(blob):
                tf[t] = tf.get(t, 0) + 1
            self._postings.append(tf)
            self._lengths.append(sum(tf.values()) or 1)
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
        self._n = len(entries) or 1
        self._avgdl = (sum(self._lengths) / self._n) if self._lengths else 1.0

    def search(self, query: str, limit: int = 5) -> list[dict]:
        q_terms = list(dict.fromkeys(_tokenize(query)))
        if not q_terms:
            return []
        scored: list[tuple[float, int]] = []
        for idx, tf in enumerate(self._postings):
            score = 0.0
            dl = self._lengths[idx]
            for t in q_terms:
                f = tf.get(t)
                if not f:
                    continue
                df = self._df.get(t, 0)
                idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
                denom = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / self._avgdl)
                score += idf * (f * (_BM25_K1 + 1)) / denom
            if score > 0:
                if self._titles[idx] in q_terms:
                    score *= 1.6
                scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[: max(1, limit)]:
            e = self.entries[idx]
            results.append(
                {
                    "ref": e["ref"],
                    "section": e["section"],
                    "title": e["title"],
                    "summary": e["summary"],
                    "snippet": _snippet(e["body"], q_terms),
                    "score": round(score, 3),
                }
            )
        return results

    def refs(self) -> list[str]:
        return [e["ref"] for e in self.entries]

    def get(self, ref: str) -> dict | None:
        e = self._by_ref.get(ref)
        if e is None:
            return None
        # Prefer the verbatim on-disk file (proves it's generated, not hardcoded).
        src = e.get("source")
        body = e["body"]
        if src and os.path.isfile(src):
            try:
                with open(src, "r", encoding="utf-8", errors="replace") as f:
                    body = f.read()
            except OSError:
                pass
        return {**e, "body": body}


def get_corpus(force: bool = False) -> Corpus:
    """Return the corpus, building+caching lazily and memoizing per process."""
    root = repo_root()
    sig = _signature(root)
    memo = _MEMO.get(str(root))
    if not force and memo and memo[0] == sig:
        return memo[1]
    corpus = Corpus(_load_entries(root, sig, force=force))
    _MEMO[str(root)] = (sig, corpus)
    return corpus


def rebuild() -> Corpus:
    """Force a fresh corpus build (used by scripts/build_corpus.py)."""
    return get_corpus(force=True)
