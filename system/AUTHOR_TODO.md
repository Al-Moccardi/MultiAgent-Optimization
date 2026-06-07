# AUTHOR_TODO — one artifact left

All §12 frozen state is committed (score_cache.json 173, score_cache_vectors.npz,
ood_queries_en.yaml, sequential + sweep + OOD result JSONs). One file remains on
the author's machine:

* `part4_dynamic_path/results/data/routing_eval_concurrent.json`
  The concurrent-chain run quoted in Table 5. One-glance check before
  committing: qa 11.07 s / qwen 9.69 s, routed k = 3.61.
  (The 06 Jun upload named `routing_eval.json` was byte-identical to the
  already-committed sequential run.)

Resolved: `random_feasible` baseline dropped from §12.4 (author decision).
