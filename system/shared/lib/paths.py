"""Minimal paths shim for the FULL codebase so part4/dynamic_lib.py runs here.
Points at this repo's actual file locations (which differ from the replication repo)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]            # repo root (parent of shared/)
SHARED = ROOT / "shared"
CORPUS = SHARED / "corpus"
PARETO = SHARED / "pareto"
_P1 = ROOT / "part1_allocation"
# canonical inputs (full-codebase locations)
AGENTS = _P1 / "config" / "agents.yaml"
CATALOG = _P1 / "config" / "catalog.zoo.yaml"
CALIBRATION = _P1 / "data" / "calib_clean_with_gold_text.yaml"
# 94-query routing-annotated calibration set (expected_agents only, no gold passages);
# used by routing_eval/ood_eval to fit per-domain F-beta thresholds. Falls back to
# the 25-query gold-text file if the 94-query file is absent.
CALIBRATION_ROUTING = _P1.parent / "data" / "manifests" / "calibration_queries_en.yaml"
PERF_TABLE = SHARED / "pareto" / "perf_table.parquet"
# quality table: prefer part4/data copy, else anywhere in repo
def _first(*cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(cands[-1])

# quality table / scorecard / test queries: check KNOWN locations only (no
# recursive glob -- a '**' glob would crawl .venv and hang).
QUALITY_TABLE = _first(ROOT / "part4_dynamic_path" / "data" / "quality_table.parquet",
                       ROOT / "shared" / "data" / "quality_table.parquet",
                       SHARED / "pareto" / "quality_table.parquet")
SCORECARD = _first(ROOT / "shared" / "data" / "quality_scorecard.csv",
                   ROOT / "data" / "manifests" / "quality_scorecard.csv",
                   SHARED / "pareto" / "quality_scorecard.csv")
TEST_QUERIES = _first(ROOT / "data" / "manifests" / "test_queries_en.yaml",
                      ROOT / "part4_dynamic_path" / "data" / "test_queries_en.yaml",
                      ROOT / "shared" / "data" / "test_queries_en.yaml")
PARAMS_B = {"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,
            "smollm2-1_7b":1.7,"gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,
            "phi3.5-mini":3.8,"mistral-7b":7.2}
MEM_BUDGET_GB = 6.99
TOK_ROUTING = 15
TOK_GENERATION = 384
def part_dirs(part_file):
    pr = Path(part_file).resolve().parents[1]
    figs, data = pr/"results"/"figures", pr/"results"/"data"
    figs.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
    return figs, data
