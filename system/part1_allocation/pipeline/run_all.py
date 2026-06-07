"""
part1_allocation/pipeline/run_all.py
====================================
End-to-end Part 1 driver.

Steps:
  1. load configs (agents, catalog, device, testset)
  2. PERFORMANCE sweep  -> perf_table.parquet            (per hardware, cheap)
  3. QUALITY sweep      -> quality_table.parquet         (once, hardware-invariant)
  4. derive MAMAP instance (Q, L, E, mu) for this hardware
  5. solve MAMAP + epsilon-constraint Pareto front
  6. assemble + save the shared bundle to shared/pareto/

Run a dry run with no models:
    python -m part1_allocation.pipeline.run_all --mode mock

Reuse an existing quality table on a new device (the whole point of the split):
    python -m part1_allocation.pipeline.run_all --mode real \
        --reuse-quality shared/pareto/quality_table.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from part1_allocation.config_loader import (load_agents, load_catalog,
                                            load_device, load_testset,
                                            load_calib, calib_to_dispatcher_samples,
                                            calib_to_specialist_samples)
from part1_allocation.factories import make_mock_factories, make_real_factories
from part1_allocation.measure import (measure_performance, measure_quality,
                                      quality_records_to_df, perf_records_to_df,
                                      save_df, load_df, assemble_bundle,
                                      make_energy_meter, detect_device, describe,
                                      print_routing_summary, gpu_self_check)
from part1_allocation.optimize import build_instance, build_frontier

HERE = Path(__file__).resolve().parents[1]
CONFIG = HERE / "config"
ROOT = HERE.parent


def main(argv=None):
    ap = argparse.ArgumentParser(description="MAMAP Part 1: allocation + Pareto front")
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--agents", default=str(CONFIG / "agents.yaml"))
    ap.add_argument("--catalog", default=str(CONFIG / "catalog.yaml"))
    ap.add_argument("--device", default=str(CONFIG / "device.yaml"),
                    help="path to device.yaml, or 'auto' to detect the device")
    ap.add_argument("--latency-sla", type=float, default=8.0,
                    help="T° per-agent latency budget (s). NOT auto-detectable.")
    ap.add_argument("--latency-metric", choices=["mean", "p95"], default="p95")
    ap.add_argument("--memory-fraction", type=float, default=None,
                    help="fraction of detected memory to use as budget M (auto mode)")
    ap.add_argument("--memory-budget-gb", type=float, default=None,
                    help="override the memory budget M entirely (auto mode)")
    ap.add_argument("--testset", default=str(HERE / "data" / "testset.example.jsonl"))
    ap.add_argument("--calib", default=str(HERE / "data" / "calib_clean.yaml"),
                    help="routing gold set; drives the dispatcher's routing-quality eval")
    ap.add_argument("--calib-fanout", action="store_true",
                    help="also fan out calib queries to their expected specialist agents")
    ap.add_argument("--out", default=str(ROOT / "shared" / "pareto"))
    ap.add_argument("--reuse-quality", default=None,
                    help="path to an existing quality_table.parquet to reuse")
    ap.add_argument("--reuse-perf", default=None,
                    help="path to an existing perf_table.parquet to reuse (skip the "
                         "performance sweep entirely). The perf table is per-device, "
                         "so only reuse one measured on THIS hardware.")
    ap.add_argument("--n-eps", type=int, default=12, help="Pareto grid resolution")
    # --- Tier-2 opt-in modelling options (defaults reproduce v3 exactly) ------
    ap.add_argument("--latency-model", choices=["worst_case", "sequential", "expected_max"],
                    default="worst_case",
                    help="specialist-stage latency model. 'worst_case' (default, "
                         "v3 §9.1): SLO holds even if all specialists activate; the "
                         "shared-model stage is a single specialist call (concurrent). "
                         "'sequential': paper Eq. 7, L_router + k*L_spec + L_synth "
                         "(use --k-activated; reproduces Fig. 7 / Table 3). "
                         "'expected_max': mean over queries of the slowest ACTIVE "
                         "specialist (architecturally-correct expected latency for "
                         "parallel specialists; needs gold active sets, present in "
                         "--gold-text runs).")
    ap.add_argument("--k-activated", type=int, default=1,
                    help="number of activated specialists on the critical path; ONLY "
                         "used by --latency-model sequential (paper sweeps k in "
                         "{1,3,5,9}). Ignored otherwise.")
    ap.add_argument("--normalize-specialists", action="store_true",
                    help="divide the specialist objective block by the number of "
                         "specialists so it shares the [0,1] scale of the router and "
                         "synth terms (Finding 4). Default off -> raw v3 sum.")
    ap.add_argument("--obj-weights", default=None,
                    help="optional per-role objective weights 'w_rt,w_syn,w_spec' "
                         "(e.g. '1,1,1'). Default: all 1.0 (the raw v3 sum).")
    ap.add_argument("--only-agents", default=None,
                    help="comma-separated agent ids to run (e.g. 'A_dispatcher'). "
                         "Lets you measure just the router with NO RAGAS judge.")
    ap.add_argument("--no-quality-dedup", action="store_true",
                    help="measure quality for every config_id (default: dedup by "
                         "(model,quant) and reuse across context variants)")
    # --- specialist (RAGAS-style) evaluation ---------------------------------
    ap.add_argument("--gold", default=None,
                    help="gold set WITH ground-truth answers (calib_clean_with_gold.yaml). "
                         "Enables specialist evaluation; fans out to specialists.")
    ap.add_argument("--corpus", default=None,
                    help="corpus text store (id->text: jsonl/json/yaml) used to resolve "
                         "gold_law_ids/gold_case_ids into context TEXT for grounding "
                         "metrics. Without it, specialists get correctness only.")
    ap.add_argument("--judge-gguf", default=None,
                    help="path to a local GGUF used as the offline scoring judge "
                         "(specialist correctness + grounding). Required for specialists.")
    ap.add_argument("--specialist-scorer", choices=["judge", "ragas"], default="judge",
                    help="'judge' = offline local judge (default); 'ragas' = the RAGAS "
                         "library (needs ragas + a configured judge_llm).")
    ap.add_argument("--gold-text", default=None,
                    help="gold set WITH passage TEXT (calib_clean_with_gold_text.yaml). "
                         "Builds the corpus from its gold_law/case_passages and gives "
                         "specialists real RAG contexts -> full grounding metrics.")
    ap.add_argument("--embedder", default="bge-m3",
                    help="embedding model for answer_relevancy + retrieval: "
                         "bge-m3 (default) | e5 | <hf-id> | hash (offline).")
    ap.add_argument("--no-logprobs", action="store_true",
                    help="disable token-level logprobs (logits_all=False) in the "
                         "real backend. 2-5x FASTER and lower memory; drops only the "
                         "confidence signal, which Stage-1 allocation does not use. "
                         "Recommended for the measurement campaign.")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="max output tokens per generation in the QUALITY sweep "
                         "(default 512). Generation time scales with this; lowering "
                         "it (e.g. 256) is the single biggest speedup for the "
                         "campaign. Latency is derived as ttft + n_tok/throughput, "
                         "so shorter outputs also mean the derived L reflects shorter "
                         "answers -- keep it large enough that answers aren't truncated.")
    ap.add_argument("--perf-repeats", type=int, default=3,
                    help="timed repetitions per config in the PERFORMANCE sweep "
                         "(default 3, median taken). Lower to 1-2 to speed up the "
                         "perf phase at some cost to TTFT/throughput stability.")
    ap.add_argument("--embed-device", choices=["auto","cpu","cuda"], default="auto",
                    help="embedder device; use cpu to keep VRAM free for judge/candidate")
    ap.add_argument("--corr-embedder", default=None,
                    help="OPTIONAL separate embedder for the correctness cosine "
                         "(Tier-3 Finding 6). Decouples correctness scoring from the "
                         "retrieval embedder so a model that parrots retrieved text "
                         "is not credited as correct. Default: reuse --embedder.")
    ap.add_argument("--retrieve", action="store_true",
                    help="use live FAISS retrieval for specialist contexts (needs built "
                         "indexes via tools.build_faiss). Default: oracle gold passages.")
    ap.add_argument("--corpus-dir", default="shared/corpus",
                    help="dir with LawCorpus_IT.faiss/.manifest etc. (for --retrieve).")
    ap.add_argument("--exclude-configs", nargs="*", default=[],
                    help="substring(s) of config_id to exclude from candidates "
                         "(e.g. 'mistral-7b' when that GGUF is the judge -- avoids self-judging).")
    ap.add_argument("--judge-n-ctx", type=int, default=4096,
                    help="context size for the offline judge (default 4096; reduce "
                         "to save VRAM, increase only if your judge prompts are long).")
    args = ap.parse_args(argv)

    all_agents = load_agents(args.agents)
    configs = load_catalog(args.catalog)
    if args.exclude_configs:
        before = len(configs)
        configs = {cid: c for cid, c in configs.items()
                   if not any(pat in cid for pat in args.exclude_configs)}
        print(f"[catalog] excluded {before - len(configs)} configs matching "
              f"{args.exclude_configs}; {len(configs)} remain")
    run_agents = all_agents
    if args.only_agents:
        sel = {x.strip() for x in args.only_agents.split(",") if x.strip()}
        run_agents = [a for a in all_agents if a.agent_id in sel]
        if not run_agents:
            raise SystemExit(f"--only-agents matched nothing in {sel}")
        # only keep configs eligible for at least one selected agent (fewer models)
        configs = {cid: c for cid, c in configs.items()
                   if any(c.context >= a.c_min for a in run_agents)}
        print(f"[only-agents] running {[a.agent_id for a in run_agents]} "
              f"over {len(configs)} eligible configs")
    if args.device == "auto":
        device = detect_device(latency_sla_s=args.latency_sla,
                               latency_metric=args.latency_metric,
                               memory_fraction=args.memory_fraction,
                               memory_budget_gb=args.memory_budget_gb)
        print(describe(device))
    else:
        device = load_device(args.device)
    samples = load_testset(args.testset)
    if args.mode == "real":
        missing = [cid for cid, c in configs.items()
                   if c.gguf_path and not Path(c.gguf_path).exists()]
        if missing:
            print(f"[warn] {len(missing)} configs have a missing GGUF file and will "
                  f"be skipped (e.g. {configs[missing[0]].gguf_path}). "
                  f"Re-run gen_catalog --download to fetch them.")
            configs = {cid: c for cid, c in configs.items() if cid not in missing}
        if not configs:
            raise SystemExit("No usable configs: all GGUF files are missing. "
                             "Run: python -m part1_allocation.tools.gen_catalog "
                             "--download ...")
    if args.calib and Path(args.calib).exists() and not args.gold:
        calib = load_calib(args.calib)
        disp = calib_to_dispatcher_samples(calib)
        samples.setdefault("A_dispatcher", []).extend(disp)
        print(f"[calib] {len(disp)} dispatcher routing queries from {Path(args.calib).name}")
        if args.calib_fanout:
            fan = calib_to_specialist_samples(calib)
            for s in fan:
                samples.setdefault(s.agent, []).append(s)
            print(f"[calib] fanned out {len(fan)} specialist queries from calib")

    # --- gold set with ground-truth answers -> specialist evaluation ----------
    corpus = None
    embedder = None
    corr_embedder = None
    if args.gold:
        from part1_allocation.config_loader import (load_gold_calib,
                                                    gold_to_dispatcher_samples,
                                                    gold_to_specialist_samples)
        gold = load_gold_calib(args.gold)
        disp = gold_to_dispatcher_samples(gold)
        samples.setdefault("A_dispatcher", []).extend(disp)
        print(f"[gold] {len(disp)} dispatcher routing queries from {Path(args.gold).name}")
        if args.corpus:
            from part1_allocation.scoring.corpus import CorpusStore
            corpus = CorpusStore.load(args.corpus)
            all_ids = [i for q in gold for i in
                       (q.get("gold_law_ids", []) + q.get("gold_case_ids", []))]
            cov = corpus.coverage(all_ids)
            print(f"[gold] corpus loaded: {len(corpus)} docs; gold-id coverage "
                  f"{cov*100:.0f}% ({len(set(all_ids))} unique gold ids)")
        spec = gold_to_specialist_samples(gold, corpus=corpus)
        for s in spec:
            samples.setdefault(s.agent, []).append(s)
        print(f"[gold] fanned out {len(spec)} specialist queries "
              f"({'WITH' if corpus else 'WITHOUT'} context text)")

    # --- gold set WITH passage TEXT -> full grounding (the complete path) ------
    if args.gold_text:
        import yaml as _yaml
        from part1_allocation.config_loader import (load_gold_calib,
                                                    gold_to_dispatcher_samples)
        from part1_allocation.scoring.corpus import (CorpusStore,
                                                     split_corpus_from_gold_text)
        from part1_allocation.scoring.scorer import Sample
        from part1_allocation.scoring.embeddings import make_embedder

        gt = load_gold_calib(args.gold_text)
        disp = gold_to_dispatcher_samples(gt)
        samples.setdefault("A_dispatcher", []).extend(disp)
        law, case = split_corpus_from_gold_text(gt)
        corpus = CorpusStore({**law, **case})
        print(f"[gold-text] {len(disp)} routing queries; corpus = {len(law)} law + "
              f"{len(case)} case = {len(corpus)} passages")

        # embedder for answer_relevancy (+ retrieval if requested)
        if args.mode == "real" or args.retrieve:
            embedder = make_embedder(args.embedder,
                                     device=None if args.embed_device=="auto" else args.embed_device)
            print(f"[embed] {args.embedder} -> {type(embedder).__name__} (dim {embedder.dim})")
            if args.corr_embedder:
                corr_embedder = make_embedder(args.corr_embedder,
                                              device=None if args.embed_device=="auto" else args.embed_device)
                print(f"[embed] correctness embedder: {args.corr_embedder} -> "
                      f"{type(corr_embedder).__name__} (dim {corr_embedder.dim}) "
                      f"[decoupled from retriever, Finding 6]")

        retriever = None
        if args.retrieve:
            from part1_allocation.scoring.retrieval import FaissRetriever
            cdir = Path(args.corpus_dir)
            retriever = FaissRetriever(embedder, corpus._t, k=5)
            retriever.add(cdir / "LawCorpus_IT.faiss", cdir / "LawCorpus_IT.manifest.json")
            retriever.add(cdir / "CaseCorpus_IT.faiss", cdir / "CaseCorpus_IT.manifest.json")
            print(f"[retrieve] live FAISS retrieval from {cdir}/ (k=5)")

        # build specialist samples: contexts from retrieval, else oracle gold passages
        n_spec = 0
        n_synth = 0
        for q in gt:
            law_ids = list(q.get("gold_law_ids", []))
            case_ids = list(q.get("gold_case_ids", []))
            if retriever is not None:
                ret_ids, contexts = retriever.retrieve_with_ids(q["text"])
            else:
                ret_ids, contexts = [], corpus.contexts_for(law_ids, case_ids)   # oracle
            for a in q.get("expected_agents", []):
                if a == "A_dispatcher":
                    continue
                samples.setdefault(a, []).append(Sample(
                    query_id=f"{q['id']}::{a}", agent=a, question=q["text"],
                    contexts=contexts, ground_truth=q.get("ground_truth_answer"),
                    expected_agents=list(q.get("expected_agents", [])),
                    difficulty=q.get("difficulty", ""), risk_level=q.get("risk_level", ""),
                    category=q.get("category", ""),
                    high_risk=bool(q.get("high_risk", False)),
                    gold_law_ids=law_ids, gold_case_ids=case_ids,
                    retrieved_ids=ret_ids))
                n_spec += 1
            # ----- SYNTHESISER sample: one per query (handles EVERY query). -----
            # Mode (a) approximation: scored against ground_truth_answer using the
            # SAME retrieved/gold contexts. In deployment the synth would also
            # receive the specialists' answers; for evaluation we approximate by
            # giving it the same evidence the specialists see and asking it to
            # produce the final user-visible answer.
            samples.setdefault("A_synth", []).append(Sample(
                query_id=f"{q['id']}::A_synth", agent="A_synth", question=q["text"],
                contexts=contexts, ground_truth=q.get("ground_truth_answer"),
                expected_agents=list(q.get("expected_agents", [])),
                difficulty=q.get("difficulty", ""), risk_level=q.get("risk_level", ""),
                category=q.get("category", ""),
                high_risk=bool(q.get("high_risk", False)),
                gold_law_ids=law_ids, gold_case_ids=case_ids,
                retrieved_ids=ret_ids))
            n_synth += 1
        print(f"[gold-text] fanned out {n_spec} specialist + {n_synth} synth queries "
              f"WITH context text ({'retrieved' if retriever else 'oracle gold passages'})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_backend = None
    if args.mode == "mock":
        backend_factory, scorer_factory = make_mock_factories(all_agents)
        peak_mem_probe = lambda b: getattr(b, "peak_mem_gb", float("nan"))  # noqa: E731
    else:
        specialists_in_run = any(not a.is_dispatcher for a in run_agents)
        if specialists_in_run and args.specialist_scorer == "judge":
            if not args.judge_gguf and embedder is None:
                raise SystemExit(
                    "Specialists are in this run but neither a judge nor an embedder "
                    "is available. Pass --judge-gguf and/or --gold-text (which "
                    "builds the embedder), restrict to the router with --only-agents "
                    "A_dispatcher, or use --specialist-scorer ragas.")
            if args.judge_gguf:
                from part1_allocation.inference.backend import LlamaCppBackend
                print(f"[judge] loading offline judge: {Path(args.judge_gguf).name}")
                judge_backend = LlamaCppBackend(gguf_path=args.judge_gguf,
                                                n_ctx=args.judge_n_ctx)
            else:
                print("[judge] no --judge-gguf -> JUDGE-FREE scoring "
                      "(embedding-based correctness/relevancy/faithfulness + "
                      "id-based context precision/recall)")
        backend_factory, scorer_factory = make_real_factories(
            all_agents, judge_backend=judge_backend, embedder=embedder,
            specialist_scorer=args.specialist_scorer, corr_embedder=corr_embedder,
            want_logprobs=not args.no_logprobs)
        peak_mem_probe = None

    # Self-check: what will specialist evaluation actually produce?
    if any(not a.is_dispatcher for a in run_agents):
        has_gt = bool(args.gold or args.gold_text)
        print("[self-check] specialist evaluation")
        if args.mode == "mock":
            print("    mode             : mock -> deterministic MockScorer (no judge)")
        else:
            print(f"    judge            : {'set' if judge_backend else ('ragas' if args.specialist_scorer=='ragas' else 'none')}")
            print(f"    embedder         : {(type(embedder).__name__) if embedder else 'none'}")
        print(f"    ground truth     : {'yes' if has_gt else 'NO -> correctness will be NaN'}")
        print(f"    context text     : {'yes' if corpus else 'NO -> faithfulness/context_* will be NaN'}")
        runnable = []
        if has_gt and (judge_backend or embedder or args.mode == "mock"):
            runnable.append("correctness")
        # answer_relevancy: works with embedder OR judge OR mock
        if embedder or judge_backend or args.mode == "mock":
            runnable.append("answer_relevancy")
        # faithfulness: judge-free path via embedder + contexts also works
        # (max cos(answer, ctx_i) -- see SpecialistScorer.score, lines 320-326)
        if corpus and (judge_backend or embedder or args.mode == "mock"):
            runnable.append("faithfulness")
        # context_precision/recall: id-set based, kept as RETRIEVER DIAGNOSTIC only
        # (not aggregated into Q in v2.1 -- written to retriever_diagnostic.csv)
        if corpus:
            runnable += ["context_precision*", "context_recall*"]
        print(f"    metrics runnable : {', '.join(runnable) if runnable else 'none'}")
        print(f"    (* = retriever diagnostic, NOT in MILP objective)")

    # --- 2. performance sweep (per hardware) ------------------------------
    if args.reuse_perf:
        perf_df = load_df(args.reuse_perf)
        # keep only configs present in the current catalogue (e.g. after filtering
        # out oversize configs), so a reused table stays consistent with --catalog.
        perf_df = perf_df[perf_df["config_id"].isin(configs.keys())].reset_index(drop=True)
        print(f"[perf] reusing {args.reuse_perf} ({len(perf_df)} rows, "
              f"filtered to current catalogue)")
    else:
        # Self-check: tell the user up-front whether memory & energy will be collected.
        gpu_self_check()
        print(f"[perf] measuring {len(configs)} configs on '{device.name}' ...")
        perf_records = measure_performance(
            configs, hardware=device.name, backend_factory=backend_factory,
            energy_meter=make_energy_meter(device.energy_backend),
            peak_mem_probe=peak_mem_probe,
            repeats=args.perf_repeats, max_tokens=min(args.max_tokens, 256),
        )
        perf_df = perf_records_to_df(perf_records)
    if len(perf_df) == 0:
        raise SystemExit(
            "All configs failed to load — likely a llama.cpp wheel that doesn't "
            "match your CPU/GPU (e.g. Windows error 0xc000001d = illegal "
            "instruction). See README 'Troubleshooting the llama.cpp wheel'.")
    save_df(perf_df, out_dir / "perf_table.parquet")
    print(f"[perf] -> {out_dir / 'perf_table.parquet'} ({len(perf_df)} rows)")
    # Did it actually work? Report how many configs got real (non-NaN) values.
    n_mem = int(perf_df["peak_mem_gb"].notna().sum())
    n_e = int(perf_df["energy_j_per_tok"].notna().sum())
    n = len(perf_df)
    mem_tag = "OK" if n_mem == n else ("PARTIAL" if n_mem else "NOT COLLECTED")
    e_tag = "OK" if n_e == n else ("PARTIAL" if n_e else "NOT COLLECTED")
    print(f"[perf] peak memory : {n_mem}/{n} configs  [{mem_tag}]")
    print(f"[perf] energy/token: {n_e}/{n} configs  [{e_tag}]")
    if n_mem:
        print(f"[perf]   memory range : "
              f"{perf_df['peak_mem_gb'].min():.2f}–{perf_df['peak_mem_gb'].max():.2f} GB")
    if n_e:
        print(f"[perf]   energy range : "
              f"{perf_df['energy_j_per_tok'].min():.3f}–{perf_df['energy_j_per_tok'].max():.3f} J/tok")
    if n_mem == 0:
        print("[perf]   note: peak_mem_gb is NaN -> the MILP memory budget will NOT bind. "
              "Check the self-check above (need NVML or nvidia-smi).")

    # --- 3. quality sweep (once; reuse if provided) -----------------------
    if args.reuse_quality:
        print(f"[quality] reusing {args.reuse_quality} (hardware-invariant)")
        quality_df = load_df(args.reuse_quality)
    else:
        print("[quality] running quality sweep (this is the expensive one) ...")
        q_records = measure_quality(run_agents, configs, samples,
                                    backend_factory, scorer_factory,
                                    routing_specialists=[a for a in all_agents
                                                         if not a.is_dispatcher],
                                    max_tokens=args.max_tokens,
                                    dedup_by_model_quant=not args.no_quality_dedup,
                                    checkpoint_path=out_dir / "quality_table.partial.parquet")
        quality_df = quality_records_to_df(q_records)
    save_df(quality_df, out_dir / "quality_table.parquet")
    print(f"[quality] -> {out_dir / 'quality_table.parquet'} ({len(quality_df)} rows)")
    # ----- quality scorecard CSV: one row per (agent, config_id) -------------
    try:
        from part1_allocation.measure.quality_csv import export_quality_csv
        agg = export_quality_csv(quality_df, out_dir / "quality_scorecard.csv")
        print(f"[quality] -> {out_dir / 'quality_scorecard.csv'} ({len(agg)} rows: "
              f"{agg['agent'].nunique()} agents x {agg['config_id'].nunique()} configs)")
    except Exception as e:
        print(f"[quality] CSV export failed: {type(e).__name__}: {e}")
    # ----- retriever diagnostic (P/R kept OUT of the optimisation objective) -
    try:
        from part1_allocation.measure.retriever_diagnostic import export_retriever_diagnostic
        retr = export_retriever_diagnostic(quality_df, out_dir / "retriever_diagnostic.csv")
        print(f"[retriever] -> {out_dir / 'retriever_diagnostic.csv'} "
              f"({retr['n_queries']} queries; "
              f"precision={retr['context_precision']:.3f} "
              f"recall={retr['context_recall']:.3f}) [diagnostic only, NOT in MILP]")
    except Exception as e:
        print(f"[retriever] CSV export failed: {type(e).__name__}: {e}")
    print_routing_summary(quality_df)

    if judge_backend is not None:
        judge_backend.close()   # free the judge's VRAM before solving

    # --- 4. derive instance ----------------------------------------------
    inst = build_instance(quality_df, perf_df, run_agents, configs, device,
                          hardware=device.name)
    print(f"[milp] instance: |A|={len(inst.agents)} |K|={len(inst.configs)} "
          f"M={inst.M}GB")

    # --- 5. Pareto front ---------------------------------------------------
    solve_kwargs: dict = {"latency_model": args.latency_model,
                          "normalize_specialists": args.normalize_specialists}
    if args.latency_model == "sequential":
        solve_kwargs["k_activated"] = args.k_activated
    if args.obj_weights:
        parts = [float(x) for x in args.obj_weights.split(",")]
        if len(parts) != 3:
            raise SystemExit("--obj-weights must be 'w_rt,w_syn,w_spec' (three numbers)")
        solve_kwargs["weights"] = tuple(parts)
    if args.latency_model != "worst_case" or args.normalize_specialists or args.obj_weights:
        kinfo = f" k_activated={args.k_activated}" if args.latency_model == "sequential" else ""
        print(f"[milp] Tier-2 options: latency_model={args.latency_model}{kinfo} "
              f"normalize_specialists={args.normalize_specialists} "
              f"weights={solve_kwargs.get('weights', '(1,1,1)')}")
    frontier = build_frontier(inst, n=args.n_eps, solve_kwargs=solve_kwargs)
    print(f"[pareto] {len(frontier)} non-dominated solutions:")
    for eps, sol in frontier:
        cap = "inf" if eps == float("inf") else f"{eps:.2f}s"
        print(f"   eps={cap:>6}  quality={sol.objective:7.3f}  "
              f"maxLat={sol.system_latency:5.2f}s  loaded={len(sol.loaded)}")

    # --- 6. assemble + save bundle ----------------------------------------
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "n_agents": len(run_agents),
        "n_configs": len(configs),
        "n_pareto": len(frontier),
        "latency_metric": device.latency_metric,
    }
    bundle = assemble_bundle(frontier, inst, configs, device,
                             hardware=device.name, manifest=manifest)
    bundle.save(out_dir)
    print(f"\n[done] shared bundle written to {out_dir}/")
    print("       -> frontier.json, ladders.json, configs.json, manifest.json")
    print("       -> quality_table.parquet, perf_table.parquet (for Part 2 replay)")


if __name__ == "__main__":
    main()
