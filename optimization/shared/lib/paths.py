"""
Shared repo-relative paths for the MAMAP-Edge replication package.

Every script in part1/part2/part3 does:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "shared" / "lib"))
    from paths import *
and then refers to PERF_TABLE, SCORECARD, etc. No absolute paths anywhere, so the
repository runs unchanged after `unzip` / `git clone`.
"""
from pathlib import Path

# repo root = parent of shared/
ROOT = Path(__file__).resolve().parents[2]

SHARED = ROOT / "shared"
DATA = SHARED / "data"
CORPUS = SHARED / "corpus"
PARETO = SHARED / "pareto"

# --- canonical inputs (read-only) ---
AGENTS = DATA / "agents.yaml"
CATALOG = DATA / "catalog_zoo.yaml"
CALIBRATION = DATA / "calibration_with_gold.yaml"
# part 4 (dynamic agentic path) inputs:
TEST_QUERIES = ROOT / "part4_dynamic_path" / "data" / "test_queries_en.yaml"
# optional, produced on the full-codebase machine (see part4_dynamic_path/data/INPUTS.md):
#   CALIBRATION_ROUTING -> the 94-query routing-annotated calibration set
#   score_cache.json    -> frozen bge-m3 relevance scores (replayed offline)
PERF_TABLE = DATA / "perf_table.parquet"
QUALITY_TABLE = DATA / "quality_table.parquet"
SCORECARD = DATA / "quality_scorecard.csv"
RETRIEVER_DIAG = DATA / "retriever_diagnostic.csv"
LAW_MANIFEST = CORPUS / "LawCorpus_IT_manifest.json"
CASE_MANIFEST = CORPUS / "CaseCorpus_IT_manifest.json"

# --- shared pareto frontiers (precomputed; regenerable by the parts) ---
PERF_FRONTIER = PARETO / "perf_capacity_frontier.csv"        # part1 output
QUALITY_ADDITIVE_FRONTIER = PARETO / "quality_additive_frontier.csv"  # part2 output
QUALITY_GATED_FRONTIER = PARETO / "quality_gated_frontier.csv"        # part3 output

# --- per-part output dirs (resolved by the caller passing __file__'s part dir) ---
def part_dirs(part_file):
    """Return (figures_dir, data_dir) for the calling part, creating them."""
    part_root = Path(part_file).resolve().parents[1]   # .../partX/src/script.py -> partX
    figs = part_root / "results" / "figures"
    data = part_root / "results" / "data"
    figs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return figs, data

# --- model metadata (params in billions), used across parts ---
PARAMS_B = {
    "smollm2-360m": 0.36, "qwen2.5-0_5b": 0.5, "llama3.2-1b": 1.2,
    "qwen2.5-1_5b": 1.5, "smollm2-1_7b": 1.7, "gemma2-2b": 2.6,
    "qwen2.5-3b": 3.0, "llama3.2-3b": 3.2, "phi3.5-mini": 3.8, "mistral-7b": 7.2,
}

# --- system constants ---
MEM_BUDGET_GB = 6.99      # usable VRAM on the RTX 4070 Laptop (8 GB)
TOK_ROUTING = 15          # dispatcher output tokens
TOK_GENERATION = 384      # specialist / synthesiser output tokens
