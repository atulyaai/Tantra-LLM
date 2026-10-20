# Research: many LoRA experts on a small CPU model

Scope: shared ~57M base + rank-8 LoRA adapters routed per request, 4-core CPU, 16 GB RAM.
Written from web searches (search-result summaries; most papers were not read in full). Confidence tags:
**[E]** well established, **[P]** plausible/transfers by analogy, **[S]** speculative. Nothing here is measured on Tantra.
Numbers for Tantra (memory, latency) are back-of-envelope and must be benchmarked before use.

## 1. Serving many LoRA adapters on CPU

**What the literature does.** S-LoRA keeps all adapters in host memory, loads only the ones the active batch needs onto the GPU,
pages adapter weights and KV cache from one pool, and uses custom kernels for heterogeneous-rank batches; the authors report up
to 4x throughput over PEFT/early vLLM [E, GPU context] ([S-LoRA paper](https://arxiv.org/pdf/2311.03285),
[project page](https://sky.cs.berkeley.edu/project/s-lora/)). The problem it solves is batching many users across many adapters
on GPU; **Tantra has batch size ~1 on CPU, so most of S-LoRA's machinery (unified paging, custom batched kernels, tensor parallelism)
is irrelevant.** The transferable ideas are: base weights shared and resident once; adapters stored cold and loaded on demand; do not merge per adapter.

**Merged vs unmerged [E by construction].**
- Unmerged: y = Wx + (alpha/r)·B(Ax). Extra cost per adapted linear is two skinny matmuls, O(r·(d_in+d_out)) vs O(d_in·d_out).
  At rank 8 and d=512 that is roughly 3% of the base matmul FLOPs per adapted layer, but several extra small kernel launches, which
  can dominate on tiny models. Switching adapters = swapping small tensors; the base is untouched. Memory per adapter is tiny
  (rank 8 on all linears of a 57M model is on the order of 1-2M params, ~2-8 MB in fp32 — verify with `sum(p.numel())`).
- Merged: W' = W + (alpha/r)BA. Zero runtime overhead, but a switch costs a full pass over adapted weights (or a second model copy,
  ~230 MB fp32 each), and un-merging in fp16/int8 accumulates rounding error. Merging into int8-quantized weights is lossy.
- llama.cpp supports exactly this split: adapters converted to GGUF, loaded without applying, and applied/scaled/switched at runtime
  via a server endpoint, without merging ([llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/10123),
  [Kaitchup write-up](https://kaitchup.substack.com/p/fast-inference-with-gguf-lora-adapters); secondary sources, API names unverified).

**Recommendation for Tantra.**
1. Keep one resident base model. Store adapters as separate small `.safetensors`/`.pt` files; do **not** eager-load all (this is what the
   implementer session is doing).
2. Use an LRU cache of N loaded adapters (N=8-16 is hundreds of MB at most; measure). Disk read of a few MB is milliseconds on SSD, so a
   cold load is cheap relative to generation; pre-warm the 2-3 most used ones.
3. Hot path: unmerged LoRA with weights held as plain tensors attached to wrapper modules; switching = pointer swap. For a pinned
   "default" expert (e.g. chat), merging into a *separate* base copy is an option only if profiling shows LoRA overhead > ~10% [P].
4. Never mutate the base in place for switching (also required by AGENTS.md: preserve weights).
5. Small-model trap [P]: with rank 8 and tiny matrices, Python/dispatch overhead of extra Linear calls can outweigh FLOPs. Measure tokens/s
   merged vs unmerged before deciding; fuse A and B as one module and adapt only q,v (or q,k,v,o) rather than every linear if latency matters.

## 2. Routing

**Two families** ([survey framing in a 2025 MoLE paper](https://arxiv.org/pdf/2508.17250)):
- *Static merging/selection* before inference (AdaMix, SMEAR, LoRAHub-style gradient-free composition).
- *Dynamic, token- or layer-level routing* (MoLoRA, LD-MoLE, hybrid routing, LoRA-Mixer)
  ([LD-MoLE](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6a39cf3b666f8bdb2223f253981f3869-Abstract-Conference.html),
  [Hybrid routing](https://ojs.aaai.org/index.php/AAAI/article/view/40383), [LoRA-Mixer](https://arxiv.org/html/2507.00029v1)).
  These train router + experts **jointly**, require all experts resident, and cost per-token compute for every active expert.

**Fit for Tantra.** Token-level MoE is the wrong tool [P, strong reasoning]: experts here are separately trained, scope-distinct adapters
(AGENTS.md), loaded lazily; joint token routing needs all experts in memory, joint training, and breaks the "one candidate adapter at a time"
workflow. Request-level routing (one decision per prompt, then run a single adapter) fits CPU, lazy loading, and independent evaluation. [P]

**Router options, cheapest first** (none verified in a paper for tiny Hindi/English/Hinglish models):
1. Rules/keywords + script detection (Devanagari vs Latin; Hinglish = Latin with Hindi tokens). Deterministic, zero-cost, testable. [E as engineering]
2. Linear classifier (logistic regression / fastText-style) on char n-gram TF-IDF. Microseconds, trainable on a few hundred labeled prompts per
   expert, CPU-trivial. [E that this works for coarse topic/intent classification; unverified at Tantra's label granularity]
3. Embedding-centroid: mean-pooled hidden states of the *base model itself* on the prompt (already computed during prefill, so free) →
   nearest centroid / small logistic head. [P] This reuses the prefill pass; router cost ≈ 0.
4. A separate small transformer classifier: probably unnecessary at this scale.

With **164 taxonomy IDs** that are specifications, not trained capabilities, route first at coarse level (maybe 6-12 groups), then
within-group; always include a **fallback = base/general chat adapter** and an abstain threshold. Route only to experts that have passed evaluation.

**Evaluating routing** [E for the metrics, standard classification practice]:
- Labeled held-out prompt set per expert, frozen split, grouped by source/problem family (paraphrases/translations in one split).
- Report per-expert precision/recall, confusion matrix, abstain/fallback rate, and *per-language* (Hindi / English / Hinglish separately).
- Report end-to-end quality delta: routed-adapter output vs base vs oracle-routed output on the same held-out prompts. Routing is
  good only if (routed − base) captures most of (oracle − base).
- Include out-of-scope and adversarial prompts (should go to fallback), and near-miss overlapping-scope pairs.
- Test on non-templated, human-phrased prompts; templated routing data will make accuracy look near 100% (see §5).

## 3. CPU inference speed (PyTorch)

| Lever | Expected gain on 4-core CPU, ~57M model | Confidence |
|---|---|---|
| Thread count = physical cores (not logical) | avoids oversubscription; sweep 1..4 | [E] |
| KV cache (incremental decode) | decode cost per token becomes O(1) in context instead of re-running whole prefix; usually the single biggest win if currently absent | [E] |
| Dynamic int8 on `nn.Linear` | docs claim int8 compute typically 2-4x faster than fp32 in hardware terms; real small-model gains often 1.2-2x, and can be slower | [E that it exists, [S] on gain] |
| `torch.compile` | helps via fusion/less overhead; compile time and Windows support are risks; small decode shapes vary | [P] |
| ONNX Runtime / llama.cpp-style | often faster per token than eager PyTorch on CPU (fused kernels, AVX2, mmap); needs export; LoRA support is the hard part | [P] |
| bf16/fp16 on CPU | only helps with AVX512-BF16/AMX; an i3 likely lacks it | [P] |

Details and sources:
- Thread guidance: settings in code override env vars; benchmark by sweeping `torch.set_num_threads`; too many threads hurt
  ([PyTorch threading notes](https://docs.pytorch.org/docs/2.3/notes/cpu_threading_torchscript_inference.html)). One thread per **physical**
  core, since hyperthreads share FMA units ([PyTorch/Intel tuning tutorial](https://PyTorch.org/tutorials/intermediate/torchserve_with_ipex)).
  A forum report shows a small model faster with 1 thread ([thread](https://discuss.pytorch.org/t/cpu-inference-is-very-slow-unnecessarily-occupies-multiple-cpu-cores/16518)).
  On decode with matrices ~512x2048, parallel overhead is significant, so test 1, 2, 4 explicitly.
- Dynamic quantization: `quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`; the old PyTorch docs say int8 is "typically 2 to 4 times faster"
  than fp32 at the hardware level ([PyTorch 1.10 quantization docs](https://docs.pytorch.org/docs/1.10/quantization.html)). The torchao
  eager tutorial shows only ~1.03x on a toy model and says speedup needs larger models ([torchao](https://docs.pytorch.org/ao/stable/eager_tutorials/first_quantization_example.html));
  a forum thread reports a *slower* quantized BERT ([thread](https://discuss.pytorch.org/t/quantized-model-of-dynamic-quantization-on-bert-tutorial-is-slower-than-original-model/99267)).
  I did not verify the current recommended API (`torch.ao.quantization` vs torchao); check installed PyTorch. Quantize the *base* only; keep
  LoRA A/B in fp32 and add them unmerged. Evaluate perplexity change before adopting. A 57M model is already small; int8 mainly saves memory/bandwidth.
- Do not use unvetted claims like "5x from compile"; none were found for this size class.

**Ordering for effort/return** [P]: (a) confirm KV cache is used in generation, (b) profile a decode step (`torch.profiler`) to see whether time is
matmul or Python overhead, (c) thread sweep, (d) int8 dynamic on base, (e) only then export to ONNX/GGUF. Exporting to llama.cpp is a bigger step:
architecture must be supported (Llama-like is easy; custom blocks are not), and the 65K tokenizer must convert. Treat as a later deployment option.

## 4. What makes 50M–300M models chat coherently

- **Pretraining tokens dominate.** TinyStories showed <10M-param models producing coherent multi-paragraph stories, but only on a
  narrow, synthetic, small-vocabulary distribution; 125M GPT-2-class models trained on general text were rarely coherent beyond a few words
  in their setup ([TinyStories](https://arxiv.org/pdf/2305.07759)) [E for the claim, narrow scope]. Lesson: **coherence comes from a
  narrow, clean, consistent distribution**, not size. Tantra's ~19M Wikipedia tokens is tiny and wide (encyclopedic, multi-language,
  65K vocab: many embedding rows get few updates). Expect poor fluency until more clean tokens exist. [P]
- Modern small models get coherence from very large token counts and curated data: SmolLM2-135M saw ~2T tokens, 360M ~4T tokens, with
  heavily filtered educational data ([model cards](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
  [360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)) [E]. Tantra is ~5 orders of magnitude below. Do not expect general chat;
  expect narrow-domain competence if the *pretraining and SFT* distributions are matched.
- **SFT needs little data only on top of a strong base.** LIMA: ~1,000 curated examples were enough for a 65B model because knowledge
  and ability come from pretraining ([LIMA](https://arxiv.org/pdf/2305.11206)) [E]. This does **not** transfer to a 57M model with 19M tokens;
  it has no "latent ability" for SFT to surface. That explains Tantra's observed "learns identity, overfits, repeats". [P, consistent with the observation]
- **LoRA vs full FT.** LoRA learns less but forgets less, and full finetuning updates have rank 10-100x higher than typical LoRA configs
  ([LoRA Learns Less and Forgets Less](https://arxiv.org/html/2405.09673v2)) [E, for 7B-class code/math]. For a tiny base, rank 8 may cap
  *capability* added; use LoRA for style/domain/format, and put new knowledge/language skill into continued pretraining. [P]
- **Decoding.** Greedy decoding causes repetition; the CTRL repetition penalty (θ≈1.2 with greedy) is a known mitigation
  ([CTRL](https://arxiv.org/pdf/1909.05858)); nucleus (top-p) sampling is from Holtzman et al. (I could not retrieve the paper in search;
  cite from memory: arXiv 1904.09751, **unverified link**) [E]. Practical default to test: temperature 0.6-0.8, top-p 0.9, repetition
  penalty 1.1-1.2 (or no-repeat-ngram 3-4), stop at EOS/turn marker, short max length. Tune on held-out prompts, not by eye on 3 samples.
  Penalties hurt structured output (JSON/code) ([note](https://dev.to/ji_ai/why-repetition-penalty-breaks-json-and-code-generation-4bbk), opinion piece), so make decoding settings per-expert.
- **Overfitting signals.** Loss on a frozen val split diverging while train drops; repeated n-grams in samples. Early-stop on val loss, low LR
  (1e-4 to 3e-4 for LoRA, lower for full FT), few epochs (1-3), dropout 0.05-0.1, mask loss to assistant tokens only. [E as standard practice]

## 5. How much own-authored data, and scaling with templates

No source gives a number for 57M Hindi/English models. Honest estimate [S], to be validated by learning curves on Tantra itself:
- Narrow, single-format expert (e.g. identity, fixed-schema tasks): ~1k–5k *distinct* examples can be enough; hundreds overfit (as seen).
- Conversational competence in one language at this scale: tens of thousands of diverse turns plus far more pretraining tokens (hundreds of M),
  not just SFT. 
- Measure, don't guess: train on 25/50/100% of data, plot held-out loss and a fixed prompt eval; if the curve is still steep at 100%, more data helps.

**Templates and diversity.**
- Duplicates and near-duplicates measurably hurt: verbatim copying of training text and wasted steps; dedup improved models
  ([Lee et al. 2022](https://research.google/pubs/deduplicating-training-data-makes-language-models-better/)) [E, web-text setting]. Templated data is
  near-duplication by construction; 9,450 template rows may be effectively far fewer *distinct* examples. [P]
- Evidence on template-augmentation specifically was not found in my searches; the following is engineering judgment [S]:
  - Measure effective diversity: distinct-n, unique-skeleton count after masking slot fillers, and near-duplicate clusters (MinHash).
  - Cap examples per skeleton (e.g. ≤3-5) and per slot value; vary skeletons themselves (syntax, register, length, politeness, code-mixing), not only fillers.
  - Group splits by skeleton family so val/test contain *unseen skeletons*; otherwise val loss measures memorization.
  - Mix in human-written (owner-written) rows at higher weight; templates fill coverage, not style.
  - Keep a small, hand-reviewed gold eval set that is never templated. AGENTS.md rules (no filler, reviewed provenance, frozen splits) apply.

## Ranked top-5 changes for Tantra

1. **Fix evaluation first: frozen, family-grouped held-out sets** (loss + fixed prompt battery, per language; skeleton-grouped for templated data).
   Everything else (routing, rank, sampling) is unmeasurable without it. Cheap, offline. [E]
2. **Decode properly before concluding the model is bad**: KV cache + temperature/top-p + repetition penalty/no-repeat-ngram + EOS stop,
   tuned on the held-out prompts. May remove much of the "repeats" symptom at zero training cost. [E/P]
3. **Lazy-load adapters with LRU cache, unmerged LoRA, one resident base** (already in progress), plus a profile of merged vs unmerged
   tokens/s and a thread sweep (physical cores). Bound memory and switch latency; keep base immutable. [E/P]
4. **Request-level routing with a cheap router and mandatory fallback**: script detection + char-n-gram linear classifier (or base-model
   prefill-embedding centroid), coarse groups first, only to evaluated experts; evaluate vs base and oracle routing. [P]
5. **Raise distinct data, not row count**: dedup/cap templated batch by skeleton, add human-written diversity, and put more clean Hindi/English/Hinglish
   tokens into continued pretraining rather than expecting SFT/LoRA to add language ability to a 19M-token base. Run a data-scaling curve
   (25/50/100%) to decide how much more is needed. [E for dedup; P/S for the rest]

Deferred until the above are done: torch.compile, ONNX/llama.cpp export, int8 dynamic quantization (try after profiling; revalidate perplexity), token-level MoE (not recommended).

## Gaps / caveats
- Search summaries only; S-LoRA, LIMA, LoRA-vs-FT, TinyStories claims come from abstracts/summaries.
- No source found for: torch.compile speedups on tiny models, ONNX Runtime LoRA support, template-augmentation effects, Hindi/Hinglish small-model data requirements.
- Nucleus sampling and Switch Transformer/LoRA Soups links were not retrieved; no claims depend on them beyond the decoding defaults above.
