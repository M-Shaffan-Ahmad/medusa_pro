# Medusa Acceleration Extensions

#### TurboQuant, Custom Trees, and Multi Minnions for Faster Local LLM Decoding

Muhammad Shaffan Ahmad (23i-0673) and Hamza Tariq (23i-0519)

<p align="center">
  <img src="artifacts/benchmarks/medusa/local_turbo_context_findings/intro.png" alt="Turbo Medusa overview" width="92%">
</p>

This repository extends the Medusa speculative-decoding framework with three
project directions:

1. **TurboQuant KV-cache compression** for longer-context Medusa decoding on
   constrained GPUs.
2. **Custom speculative tree sizing** so the verifier does not always pay for a
   full 63/64-node tree.
3. **Multi Minnions**, a set of small niche-specific Medusa head packs that can
   be routed by task type.

The final report and presentation are included in the repo:

| Artifact | File |
| --- | --- |
| Final report | [turbo_medusa_minions.pdf](artifacts/benchmarks/medusa/local_turbo_context_findings/turbo_medusa_minions.pdf) |
| Presentation PDF | [presentation.pdf](presentation.pdf) |
| Presentation HTML | [presentation.html](presentation.html) |
| TinyLlama optimisation report | [tinyllama_medusa_optimisation_report.pdf](tinyllama_medusa_optimisation_report.pdf) |

## Motivation

Autoregressive LLM decoding is serial: every generated token normally needs a
full target-model step. Medusa improves this by adding lightweight heads that
draft multiple future tokens and verify a token tree in parallel. At longer
context windows, however, the KV cache grows linearly with context length, tree
verification can become unnecessarily expensive, and one generic head pack may
not be equally good for coding, chat, summarization, and reasoning.

This project targets those bottlenecks by reducing long-context KV memory
pressure, calibrating Medusa tree size, and training small specialized head
packs while keeping the base model unchanged.

## Main Findings

| Area | Finding |
| --- | --- |
| TurboQuant KV cache | Reduced KV-cache size by roughly 3.6x and enabled 32k context on a 6 GB RTX 3060 Laptop GPU where base Medusa OOMed. |
| Raw throughput | Current TurboQuant path stores compressed KV but decodes temporary dense K/V before attention, so it is a capacity win first rather than a raw TPS win. |
| Custom trees | 24-node/custom trees beat full 63/64-node trees in the final sweeps for TinyLlama and Llama-3.2. |
| Multi Minnions | Specialized head packs are promising: the coding-specialized Llama-3.2 heads reached 40.4% top-1 on head 1 after under 2 hours of RTX 3080 training. |
| Reliability boundary | Exact verifier acceptance remains the correctness boundary; QJL and pruning are treated as planning signals, not final acceptance rules. |

## SOTA Context

| Work | Core idea | Relevance to this project | Gap targeted here |
| --- | --- | --- | --- |
| Speculative Decoding / Speculative Sampling | Use a cheaper draft model and verify several tokens with the target model. | Establishes draft-and-verify as the main latency reduction pattern. | Requires a good draft model and can add model-management cost. |
| SpecInfer | Builds and verifies a speculative token tree. | Closest systems-level ancestor of Medusa-style tree verification. | Tree size should be hardware and acceptance aware. |
| Medusa | Adds multiple draft heads directly to the target model and verifies candidate trees. | Base system used in this repo. | Fixed/default trees and generic heads are not always optimal. |
| Hydra | Makes draft heads sequentially dependent to improve Medusa-style speculation quality. | Supports the idea that head architecture and conditioning matter. | This project explores domain specialization instead. |
| EAGLE | Drafts in feature space for stronger speculative acceptance. | Shows stronger draft representations can outperform simple heads. | More complex drafting can be harder to train and deploy locally. |
| PagedAttention / vLLM | Manages KV cache blocks to reduce serving fragmentation. | Orthogonal memory-management direction. | Does not compress numerical KV vectors directly. |
| KVQuant / KIVI | Sub-4-bit KV-cache quantization using outlier-aware or asymmetric layouts. | Strong long-context KV compression baselines. | Accuracy and speed depend on metadata layout, calibration, and custom kernels. |
| PolarQuant / QJL / TurboQuant | Polar transforms, 1-bit residual correction, and near-optimal online vector quantization. | Most direct SOTA context for this implementation. | Reported speedups require fused compressed-attention kernels; this repo still decodes temporary dense K/V. |

## TurboQuant KV Cache

The implemented TurboQuant path stores KV vectors in compressed form using
random rotation, scalar quantization, per-token scale metadata, and a 1-bit QJL
residual correction for keys.

```text
new fp16 K/V -> encode once -> store compressed cache
attention read -> decode temporary K/V view -> verify/generate
compressed cache remains stored
```

TurboQuant therefore provides a capacity improvement first. It lowers peak GPU
allocation and allows longer context windows than base Medusa on the same
hardware. The expected throughput gain depends on removing the temporary dense
decode path with fused compressed-attention kernels.

<p align="center">
  <img src="artifacts/benchmarks/medusa/local_turbo_context_findings/throughput_vs_context.png" alt="Throughput vs context" width="82%">
</p>

<p align="center">
  <img src="artifacts/benchmarks/medusa/local_turbo_context_findings/speed_ratio_vs_context.png" alt="TurboQuant speed ratio vs context" width="82%">
</p>

<p align="center">
  <img src="artifacts/benchmarks/medusa/local_turbo_context_findings/memory_vs_context.png" alt="Memory vs context" width="82%">
</p>

### Long-Context Measurements

The local measurements were run on an RTX 3060 Laptop GPU with 6 GB VRAM and
32 generated tokens per context point. The 32k base Medusa run OOMed during
initial prompt prefill.

| Context | Base TPS | TurboQuant b4 TPS | Speed Ratio | Base Peak MB | Turbo Peak MB | Base KV Cache MB | Turbo KV Cache MB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 35.37 | 14.75 | 0.417 | 3043.0 | 3010.8 | 34.2 | 9.3 |
| 2,048 | 33.88 | 13.86 | 0.409 | 3136.8 | 3081.1 | 64.8 | 17.7 |
| 4,096 | 24.98 | 10.84 | 0.434 | 3329.3 | 3227.5 | 127.6 | 34.9 |
| 8,192 | 14.99 | 7.45 | 0.497 | 3709.0 | 3512.8 | 253.1 | 69.2 |
| 16,384 | 8.10 | 4.49 | 0.554 | 4466.8 | 4093.0 | 504.3 | 137.9 |
| 32,768 | OOM | 2.20 | N/A | OOM | 5227.5 | 1005.0 | 274.8 |

At 16k context, base Medusa and TurboQuant both fit, but TurboQuant used about
374 MB less peak allocation. At 32k context, base Medusa OOMed while
TurboQuant b4 completed at 2.20 TPS.

## Custom Tree Size

The default full Medusa tree is not universally best. Larger trees increase
candidate coverage, but they also increase verification work. If head quality
or task predictability is not high enough, many extra nodes do not translate
into accepted tokens.

<p align="center">
  <img src="artifacts/benchmarks/medusa/local_turbo_context_findings/tree.png" alt="Custom tree-size summary" width="82%">
</p>

### Tree Sweep Summary

| Model / setting | 64-node/full tree max TPS | 24-node/custom tree max TPS | Finding |
| --- | ---: | ---: | --- |
| TinyLlama final sweep | 121.79 TPS | **151.17 TPS** | The 24-node tree gave the best observed max TPS, about **1.24x** over the full-tree run. |
| Llama-3.2 final sweep | 53.13 TPS | **59.60 TPS** | The 24-node tree gave the best observed max TPS, about **1.12x** over the full-tree run. |
| Quick calibration sweep | 108.40 TPS | **129.04 TPS** | The 24-node setting also won in the quick calibration run, about **1.19x** over the full-tree max. |

<p align="center">
  <img src="artifacts/benchmarks/medusa/local_turbo_context_findings/llma3.2.jpeg" alt="Llama 3.2 tree sweep" width="48%">
  <img src="artifacts/benchmarks/medusa/local_turbo_context_findings/tiny.jpeg" alt="TinyLlama tree sweep" width="48%">
</p>

The better policy is to sweep candidate limits such as 8, 16, 24, 32, 40, and
63/64, compare TPS, accepted tokens per step, and prefix match, and treat the
full tree as a fallback rather than the default answer.

## Multi Minnions

Medusa heads are small compared with the base model, so they are cheap to train
and easy to swap. Multi Minnions means training several small speculative head
packs, each specialized for a narrow workload.

```text
base LLM
  + coding Minnion heads
  + chat Minnion heads
  + summarization Minnion heads
  + reasoning Minnion heads
  + domain/tool-use Minnion heads
```

At inference time, a lightweight router can select a head pack based on prompt
type, or the user can explicitly choose one. The base model remains unchanged.

Specialized heads help because task-specific data lowers the entropy of the
next-token distribution seen by the heads. The local coding-specialized
Llama-3.2 Medusa heads reached a validation score of 0.410 on the coding
corpus with these head top-1 accuracies:

```text
head1 40.4%, head2 22.1%, head3 14.7%, head4 10.7%
```

The earlier mixed Llama-3.2 head run logged a score of 0.174, which supports
the direction that niche heads can become much stronger on their target domain.

## TinyLlama Medusa Path

The TinyLlama report targets `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with a local
four-head Medusa configuration. The implementation uses tree decoding and
compares standard autoregressive generation against Medusa generation,
reduced-tree Turbo modes, Triton-assisted kernels, QJL pruning signals,
KV-cache compression, and self-distilled Medusa head training.

| Item | Value |
| --- | --- |
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Local Medusa config | `TinyLlama-1.1B-Chat-v1.0-4heads/` |
| Medusa heads | 4 |
| Medusa layers | 1 |
| Decoding strategy | Tree decoding |
| Reliable verifier | Exact base-model acceptance |

The key conclusion is that Medusa performance is governed by accepted tokens
per decoding step, not tree size alone. Reduced trees can lower verifier work,
but they only improve end-to-end speed when they preserve enough accepted
prefix depth.

## Repository Map

| Path | Purpose |
| --- | --- |
| `medusa/model/medusa_model.py` | Medusa model wrapper and generation path. |
| `medusa/model/medusa_choices.py` | Full and reduced speculative tree definitions. |
| `medusa/model/kv_cache.py` | FP16, packed QJL, TurboVQ, hybrid, and polar KV-cache experiments. |
| `medusa/model/triton_kernels.py` | Triton fast paths for QJL scoring, selection, cache operations, and argmax. |
| `bench_transformers_base.py` | Hugging Face autoregressive baseline benchmark. |
| `bench_medusa.py` | Medusa benchmark entrypoint. |
| `bench_comm_turbo.py` | Communication/TurboQuant and tree-sweep benchmark harness. |
| `run_batch_benchmark.py` | Batch prompt-suite benchmark driver. |
| `train_tinyllama_medusa_heads.py` | Self-distillation training path for TinyLlama Medusa heads. |
| `kaggle_mix_and_train_medusa.py` | Mixed dataset training recipe for Medusa heads. |
| `artifacts/benchmarks/medusa/local_turbo_context_findings/` | Final report, plots, and generated benchmark artifacts. |

## Reproducing Benchmarks

Install the repo in editable mode:

```bash
pip install -e .
```

Run a plain autoregressive baseline:

```bash
python bench_transformers_base.py \
  --model-dir TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --out-csv base_transformers_benchmark.csv \
  --target-new-tokens 160 \
  --prompt-suite mixed
```

Run Medusa/Turbo tree sweeps:

```bash
python bench_comm_turbo.py \
  --model-dir TinyLlama-1.1B-Chat-v1.0-4heads \
  --out-csv comm_turbo_benchmark.csv \
  --target-new-tokens 160 \
  --prompt-suite mixed \
  --choice-sweep 24,32
```

Run the batch benchmark driver:

```bash
python run_batch_benchmark.py
```

Generated CSV files are the source of truth for exact averaged throughput,
prefix-match, acceptance, and memory values.

## Final Configuration Recommendation

| Component | Recommendation |
| --- | --- |
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Medusa heads | Local four-head Medusa setup |
| Generation mode | Greedy Medusa generation with exact verifier acceptance |
| Main tree | Full Medusa tree for conservative reliability |
| Reduced tree | Report and deploy as a calibrated Turbo ablation |
| QJL | Use only as a pruning/ranking signal before exact verification |
| KV compression | Report as a capacity-focused ablation until fused compressed attention is added |
| Training | Freeze base model; train Medusa heads with self-distilled greedy future tokens |

## Takeaways

1. TurboQuant works as a KV capacity extension for base Medusa and enabled 32k
   context locally where the baseline OOMed.
2. Raw speedup is not visible yet because the implementation still decodes
   temporary dense K/V and lacks fused compressed attention.
3. Tree size should be calibrated per model and workload; full 63/64-node trees
   are not consistently optimal.
4. Multi Minnions is the strongest training-side idea: keep the base model
   fixed, train small specialized head packs, and route prompts to the best
   niche.

## References

- Leviathan, Kalman, Matias. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), 2023.
- Chen et al. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), 2023.
- Miao et al. [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781), 2023.
- Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
- Cai et al. [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774), 2024.
- Li et al. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077), 2024.
- Ankner et al. [Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding](https://arxiv.org/abs/2402.05109), 2024.
- Hooper et al. [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079), 2024.
- Liu et al. [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750), 2024.
- Zandieh et al. [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482), 2024.
- Han et al. [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617), 2025.
- Zandieh et al. [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874), 2025.

## Citation

This project builds on the original Medusa framework:

```bibtex
@article{cai2024medusa,
  title   = {Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads},
  author  = {Tianle Cai and Yuhong Li and Zhengyang Geng and Hongwu Peng and Jason D. Lee and Deming Chen and Tri Dao},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2401.10774}
}
```
