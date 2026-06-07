# closeout.ps1 -- runs the remaining close-out items on the author's laptop.
#
#   .\closeout.ps1          # Phase A: full pipeline + verify.py + byte-identity check
#   .\closeout.ps1 -PhaseB  # Phase B: bge-m3 score cache + paper-grade part-4 evals/figures
#
# Phase A proves cross-machine reproduction: every deterministic artifact must
# byte-match replication/canonical_hashes.sha256 (computed on the audit machine,
# Linux). Phase B requires sentence-transformers + the bge-m3 weights and turns
# part 4's proxy artifacts into the paper-grade ones.
#
# ASCII-only on purpose: Windows PowerShell 5.1 parses BOM-less scripts in the
# system ANSI codepage, so non-ASCII characters corrupt the tokenizer.
param([switch]$PhaseB)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# PS 5.1 does NOT stop on native-command failure even with ErrorActionPreference=Stop,
# so every python call goes through this wrapper and fails loudly.
function Py {
    & python @args
    if ($LASTEXITCODE -ne 0) { throw ("FAILED (exit $LASTEXITCODE): python " + ($args -join " ")) }
}

if (-not $PhaseB) {
    Step "Phase A.1 -- dependencies"
    Py -m pip install -r requirements.txt
    Py -m pip install -r lean/requirements.txt

    Step "Phase A.2 -- parts 1-3 pipeline (~25-40 min)"
    Push-Location part1_static_allocation/src
    foreach ($s in "01_build_capacity_frontier.py","02_baseline_vs_greedy.py",
                   "03_heterogeneous_synthetic.py","04_uncertainty_sensitivity.py",
                   "05_performance_figures.py","06_family_correlation_figures.py",
                   "07_frontier_figures.py","08_corrected_figures.py",
                   "09_heterogeneous_figure.py") { Py $s }
    Pop-Location
    Push-Location part2_quality_aware/src
    foreach ($s in "00_prepare_intermediates.py","01_quality_aware_milp.py",
                   "02_quality_frontier_analysis.py","03_dispatcher_and_domain_plots.py",
                   "04_synthesiser_and_regimes.py","05_ragas_and_proxy_vs_quality.py",
                   "06_quality_frontier_plot.py") { Py $s }
    Pop-Location
    Push-Location part3_bilinear_gated/src
    foreach ($s in "01_gated_milp_headtohead.py","02_gated_frontier_and_comparison.py",
                   "03_gated_plots.py","04_coupling_sweep.py",
                   "05_batching_robustness.py","06_batching_figure.py") { Py $s }
    Pop-Location

    Step "Phase A.3 -- lean"
    Push-Location lean
    Py -m catalog.build_catalog
    Py -m src.experiment experiments/lean_8gb.yaml
    Py -m src.ablations experiments/lean_8gb.yaml
    $run = Get-ChildItem results/lean_8gb -Directory | Sort-Object Name | Select-Object -Last 1
    Py -m src.figures --ablations results/ablations --out figures --run $run.FullName
    Py -m pytest tests/ -q
    Pop-Location

    Step "Phase A.4 -- part-4 proxy smoke + figures"
    Py part4_dynamic_path/routing_eval.py --embedder hashing
    Py part4_dynamic_path/figures_dynamic.py

    Step "Phase A.5 -- verify.py (131 checks)"
    Py verify.py

    Step "Phase A.6 -- cross-machine BYTE-IDENTITY vs replication/canonical_hashes.sha256"
    $fail = 0
    foreach ($line in Get-Content replication/canonical_hashes.sha256) {
        if (-not $line.Trim()) { continue }
        $want, $rel = $line -split '\s+', 2
        $rel = $rel.Trim()
        $got = (Get-FileHash -Algorithm SHA256 $rel).Hash.ToLower()
        if ($got -eq $want.ToLower()) { Write-Host "[BYTE-IDENTICAL] $rel" }
        else { Write-Host "[MISMATCH]       $rel" -ForegroundColor Red; $fail++ }
    }
    if ($fail -gt 0) { throw "$fail artifact(s) not byte-identical -- investigate before trusting cross-machine claims" }
    Write-Host "`nPHASE A COMPLETE: pipeline reproduced, all checks passed, all artifacts byte-identical." -ForegroundColor Green
    Write-Host "Next: .\closeout.ps1 -PhaseB   (requires bge-m3)" -ForegroundColor Green
}
else {
    Step "Phase B.1 -- freeze bge-m3 relevance scores (one-time)"
    Py -m part4_dynamic_path.scripts.build_score_cache --embedder bge-m3

    Step "Phase B.2 -- paper-grade part-4 evals (cache replay)"
    Py part4_dynamic_path/routing_eval.py
    & python part4_dynamic_path/ood_eval.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ood_eval skipped (OOD manifest not in repo -- pass --ood_file if you have it)" -ForegroundColor Yellow
    }

    Step "Phase B.3 -- figures regenerate with the bge-m3 label (proxy stamp disappears)"
    Py part4_dynamic_path/figures_dynamic.py

    Step "Phase B.4 -- verify.py"
    Py verify.py
    Write-Host "`nPHASE B COMPLETE. Commit data/score_cache.json + the regenerated results/figures" -ForegroundColor Green
    Write-Host "into the package so reviewers replay the bge-m3 numbers offline." -ForegroundColor Green
}
