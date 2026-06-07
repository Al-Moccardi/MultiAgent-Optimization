"""
part1_allocation/tools/gen_catalog.py
=====================================
Build a catalog from the SLM-zoo registry, robustly against flaky networks.

Strategy (important): we do NOT depend on the flaky HuggingFace *listing* API.
GGUF filenames follow two known conventions, so we DERIVE candidate names and
download them directly (downloads are reliable even when metadata calls reset):

  * bartowski / standard : "<RepoBase>-<QUANT>.gguf"  (case preserved)
  * Qwen official        : "<repobase>-<quant>.gguf"  (all lowercase)

Each download is retried with backoff on transient errors (e.g. WinError 10054).
A 404 (wrong name) skips to the next candidate. The catalog is written with ONLY
the files that actually exist on disk, so the downstream run can never crash on a
missing GGUF.

Examples
--------
python -m part1_allocation.tools.gen_catalog --quants Q4_K_M --contexts 4096 \
    --max-params 7.2 --download --out part1_allocation/config/catalog.zoo.yaml

# offline (no network): emit best-guess filenames you fill in / download yourself
python -m part1_allocation.tools.gen_catalog --quants Q4_K_M --contexts 4096
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_REGISTRY = HERE / "slm_zoo.yaml"

# be a bit more patient with metadata calls
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


def _candidate_filenames(repo: str, quant: str) -> list[str]:
    """Two naming conventions, in order of likelihood."""
    base = repo.split("/")[-1]
    if base.lower().endswith("-gguf"):
        base = base[:-5]
    cands = [
        f"{base}-{quant}.gguf",                  # bartowski etc. (case preserved)
        f"{base.lower()}-{quant.lower()}.gguf",  # Qwen official (lowercase)
    ]
    out, seen = [], set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _entry_not_found_cls():
    try:
        from huggingface_hub.utils import EntryNotFoundError
        return EntryNotFoundError
    except Exception:
        return ()  # nothing matches -> all errors treated as transient


def _download_first(repo: str, candidates: list[str], models_dir: Path,
                    attempts: int = 4, backoff: float = 2.0):
    """Try each candidate filename, retrying transient errors. Returns
    (filename, local_path) on success, else (None, None)."""
    from huggingface_hub import hf_hub_download
    NotFound = _entry_not_found_cls()
    dead: set[str] = set()
    last_err = None
    for attempt in range(attempts):
        for fname in candidates:
            if fname in dead:
                continue
            try:
                path = hf_hub_download(repo_id=repo, filename=fname,
                                       local_dir=str(models_dir))
                return fname, path
            except NotFound:
                dead.add(fname)            # definitively wrong name, don't retry
            except Exception as e:
                last_err = e               # transient -> retry on next pass
        if len(dead) == len(candidates):
            break                          # every candidate is a 404
        if attempt < attempts - 1:
            time.sleep(backoff * (attempt + 1))
    if last_err:
        print(f"   ! gave up after {attempts} attempts: {last_err}")
    return None, None


def _resolve_via_list(repo: str, quant: str, attempts: int = 3, backoff: float = 2.0):
    """Fallback only: query the (flaky) listing API with retries."""
    from huggingface_hub import list_repo_files
    for attempt in range(attempts):
        try:
            files = list_repo_files(repo)
            cands = [f for f in files
                     if f.lower().endswith(".gguf") and quant.lower() in f.lower()]
            single = [c for c in cands if "-of-" not in c.lower()]
            cands = single or cands
            return sorted(cands, key=len)[0] if cands else None
        except Exception as e:
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
            else:
                print(f"   ! list failed: {e}")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate a catalog from the SLM zoo")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--quants", default="Q4_K_M",
                    help="comma-separated quant labels, e.g. 'Q4_K_M,Q8_0'")
    ap.add_argument("--contexts", default="4096",
                    help="comma-separated context lengths, e.g. '4096,8192'")
    ap.add_argument("--max-params", type=float, default=8.0,
                    help="drop models larger than this (GB-ish) to fit the device")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default="part1_allocation/config/catalog.zoo.yaml")
    ap.add_argument("--download", action="store_true",
                    help="download GGUFs (recommended); catalog gets only real files")
    args = ap.parse_args(argv)

    quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    models_dir = Path(args.models_dir)
    if args.download:
        models_dir.mkdir(parents=True, exist_ok=True)

    reg = yaml.safe_load(Path(args.registry).read_text())
    out_models, skipped = [], []

    for m in reg["models"]:
        if m.get("params_b", 0) > args.max_params:
            continue
        repo = m["gguf_repo"]
        print(f"[{m['key']}] {m.get('params_b','?')}B  repo={repo}")
        quantizations = []
        for q in quants:
            cands = _candidate_filenames(repo, q)
            if args.download:
                fname, _path = _download_first(repo, cands, models_dir)
                if fname is None:                       # last resort: ask the API
                    disc = _resolve_via_list(repo, q)
                    if disc:
                        fname, _path = _download_first(repo, [disc], models_dir)
                if fname is None:
                    print(f"   x {q}: could not obtain a file, skipping this quant")
                    skipped.append((m["key"], q))
                    continue
                print(f"   {q} -> {fname}")
                quantizations.append({"label": q,
                                      "gguf_path": f"{args.models_dir}/{fname}"})
            else:
                # offline: best-guess filename (verify / download yourself)
                quantizations.append({"label": q,
                                      "gguf_path": f"{args.models_dir}/{cands[0]}"})
        if not quantizations:
            skipped.append((m["key"], "ALL"))
            continue
        out_models.append({
            "key": m["key"], "name": m.get("name", m["key"]),
            "hf_id": m.get("hf_id", ""), "params_b": m.get("params_b"),
            "contexts": contexts, "quantizations": quantizations,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml.safe_dump({"models": out_models}, sort_keys=False))
    n_cfg = sum(len(m["quantizations"]) * len(m["contexts"]) for m in out_models)
    print(f"\n[done] {len(out_models)} models -> {n_cfg} configs -> {args.out}")
    if skipped:
        print(f"[warn] skipped (model,quant): {skipped}  "
              f"(re-run to retry — already-downloaded files are reused)")
    if not args.download:
        print("[note] offline mode: gguf_path are best-guesses; run with --download "
              "to fetch real files and write verified paths.")


if __name__ == "__main__":
    main()
