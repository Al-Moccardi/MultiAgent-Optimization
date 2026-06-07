"""
part1_allocation/scoring/corpus.py
==================================
The CORPUS TEXT STORE: maps a document id -> its text.

Why this exists
---------------
The gold set (calib_clean_with_gold.yaml) references the relevant documents by
**id** (e.g. law `IT_CC_art_602`, case `ECLI:IT:TRGO:2018:431`). The FAISS files
hold only *vectors* (no text, no id map). But RAGAS-style grounding metrics
(faithfulness, context precision/recall) need the **text** of the contexts.

So the missing piece to enable the full specialist evaluation is a file that maps
each gold id to its text. Supply it via `--corpus` and this store resolves a
sample's gold ids into the `contexts` list used for scoring (an *oracle-retrieval*
evaluation: it scores generation quality given the correct documents; swap in a
live FAISS retriever later to also score retrieval).

Accepted formats (auto-detected by extension)
---------------------------------------------
* JSONL : one object per line, {"id": "...", "text": "..."}   (recommended)
* JSON  : a single object {"id": "text", ...}  OR a list of {"id","text"}
* YAML  : {documents: [{id, text}, ...]}  OR  {id: text, ...}

Missing ids are skipped with a warning (so a partial corpus still runs).
"""
from __future__ import annotations

import json
from pathlib import Path


class CorpusStore:
    def __init__(self, id2text: dict[str, str]):
        self._t = dict(id2text)

    # -- loading ----------------------------------------------------------- #
    @staticmethod
    def load(path: str | Path) -> "CorpusStore":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"corpus store not found: {p}")
        suf = p.suffix.lower()
        if suf in (".jsonl", ".ndjson"):
            return CorpusStore(_load_jsonl(p))
        if suf == ".json":
            return CorpusStore(_normalize(json.loads(p.read_text(encoding="utf-8"))))
        if suf in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            # gold-text yaml? gather passages from queries.
            if isinstance(data, dict) and "queries" in data:
                law, case = split_corpus_from_gold_text(data["queries"])
                return CorpusStore({**law, **case})
            return CorpusStore(_normalize(data))
        return CorpusStore(_load_jsonl(p))

    # -- access ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._t)

    def get(self, doc_id: str) -> str | None:
        return self._t.get(doc_id)

    def resolve(self, ids: list[str], *, warn: bool = True) -> tuple[list[str], list[str]]:
        """Return (texts_found, ids_missing) for a list of doc ids."""
        texts, missing = [], []
        for i in ids:
            t = self._t.get(i)
            if t is None:
                missing.append(i)
            else:
                texts.append(t)
        if missing and warn:
            print(f"[corpus] {len(missing)} ids not found (e.g. {missing[:3]})")
        return texts, missing

    def contexts_for(self, gold_law_ids: list[str], gold_case_ids: list[str]) -> list[str]:
        """Oracle contexts = texts of the gold law + case ids (laws first)."""
        texts_law, _ = self.resolve(list(gold_law_ids), warn=False)
        texts_case, _ = self.resolve(list(gold_case_ids), warn=False)
        return texts_law + texts_case

    def coverage(self, ids: list[str]) -> float:
        """Fraction of ids present (diagnostic for the self-check)."""
        if not ids:
            return 1.0
        have = sum(1 for i in ids if i in self._t)
        return have / len(ids)


def split_corpus_from_gold_text(queries: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Collect unique law and case passages {id: text} from a gold-text yaml's
    queries (fields gold_law_passages / gold_case_passages, each [{id, text}])."""
    law: dict[str, str] = {}
    case: dict[str, str] = {}
    for q in queries:
        for p in q.get("gold_law_passages", []):
            law[str(p["id"])] = str(p.get("text", ""))
        for p in q.get("gold_case_passages", []):
            case[str(p["id"])] = str(p.get("text", ""))
    return law, case


def _load_jsonl(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[str(r["id"])] = str(r.get("text", r.get("content", "")))
    return out


def _normalize(obj) -> dict[str, str]:
    """Accept {id: text}, [{id,text}], or {documents:[...]} -> {id: text}."""
    if isinstance(obj, dict) and "documents" in obj:
        obj = obj["documents"]
    if isinstance(obj, dict):
        return {str(k): str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        out = {}
        for r in obj:
            out[str(r["id"])] = str(r.get("text", r.get("content", "")))
        return out
    raise ValueError("unrecognised corpus format")
