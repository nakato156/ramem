# Exact proposal for E01–E05

Do not execute these steps until the M0–M2 infrastructure review is accepted.

1. **Data gate:** pin dataset revisions and licenses in `data/datasets_manifest.yaml`, download only
   authorized splits into immutable `data/raw/`, compute SHA-256 checksums, and unblock entries only
   after verification. Build a small development slice first; keep MLQA, XQuAD and every test-only
   split outside training and prompt selection.
2. **E00:** validate evaluators against reviewed rows sampled from authorized development data and
   preserve their source IDs. Block later experiments until each evaluator passes its sanity cases.
3. **E01:** install the official Gemma 3 1B Instruct runtime behind the `Generator` protocol, use the
   tokenizer's official chat template, freeze prompt/temperature/token limit, and persist per-example
   predictions and resource metrics. No fine-tuning.
4. **E02:** implement BM25, EmbeddingGemma dense retrieval and RRF adapters over the same normalized
   MIRACL Spanish development corpus. Compare Recall@1/5/10/20, nDCG@10, MRR@10, p50/p95 latency,
   RAM and index size with fixed queries and ordering.
5. **E03:** rebuild only the dense index at 768, 256 and 128 dimensions. Accept 256 by default if its
   relative nDCG@10 loss versus 768 is at most two points; otherwise retain 768.
6. **E04:** compare 128/256/512-token chunks at 10% overlap, then test overlap only for the winning
   size. Add parent-child chunking as a separate final comparison; avoid the full Cartesian product.
7. **E05:** on the development set only, sweep final `k` over 3/5/8 and context budgets over
   1K/2K/4K. Choose the smallest context on the quality/latency Pareto frontier.

Every run writes the resolved config, commit, environment, seed, dataset hashes, index metadata,
predictions and timings. Run final test splits once only after the release candidate is frozen.
