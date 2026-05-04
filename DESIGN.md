# BEANS-Next — Design Document (v2.6)

**Status:** Draft v2.6 · supersedes v2.5 (rename `beans-next` → **BEANS-Next**; scope narrowed to bioacoustics-only; first deliverable is reproducing BEANS-Zero on NatureLM-audio 1.0 and 1.1 side-by-side; `earthspecies/beans-zero` coexists as the legacy package; clean break from the Callable model API). **What this is:** a bioacoustics audio-language-model benchmark library from Earth Species Project. Positions as the successor to BEANS-Zero, with published model-comparison results as the north-star deliverable. **Scope:** bioacoustics only — species classification, detection, captioning, lifestage, call-type, and related tasks. General-audio benchmarks (LibriSpeech, CoVoST, MMAU, etc.) are explicitly out of scope. **Reintegration:** Standalone library \+ bridge package (default); absorb-into-`esp_research` kept open as alternative.

---

## 1\. Framing

**BEANS-Next** is Earth Species Project's second-generation public bioacoustics benchmark. It extends BEANS-Zero (Robinson et al., ICLR 2025; `earthspecies/beans-zero`) with a proper benchmark framework: HTTP-based model serving, paper-reproduction workflows, two-layer caching, BEANS-Zero task coverage plus new bioacoustic tasks, and Tier-1 support for NatureLM-audio 1.0 and 1.1 side-by-side.

The first deliverable — non-negotiable — is **reproducing BEANS-Zero published numbers for NatureLM-audio 1.0 and 1.1 side-by-side** via the benchmark. Nothing else ships before this works.

### 1.1 Positioning vs existing public packages

BEANS-Next coexists with the existing `earthspecies/beans-zero` package rather than absorbing or depending on it. They share no code. Users pick one:

- **`earthspecies/beans-zero` (v1.0.0, April 2025\)**: in-process `Callable` model interface, `run_benchmark()` Python API, `beans-fetch`/`beans-info`/`beans-evaluate` CLIs. Stable, published, works today. Good for one-off evaluations of a single local model.  
- **BEANS-Next**: HTTP-based model serving, multi-model/multi-task run configs, paper-reproduction pipelines, caching, streaming, resumability, launcher ecosystem. Aimed at sustained benchmarking workflows — paper reproduction, model comparison, leaderboard maintenance.

The legacy `beans-zero` package is **not a dependency and not a migration target**. Users with `Callable`\-style models rewrite them as HTTP launchers (one per model) using BEANS-Next's `examples/servers/` as a template. This is a clean break by design: maintaining `Callable` compat would pull the HTTP-only architecture out of shape for a shrinking user base.

BEANS-Next does share the dataset: both packages load from `EarthSpeciesProject/BEANS-Zero` on HuggingFace.

### 1.2 Scope boundary — bioacoustics only

BEANS-Next is a bioacoustics benchmark. Tasks in scope:

- **Species classification** (single-label, including unseen species / genus / family at common / scientific / taxonomic levels).  
- **Detection** (multi-label: comma-separated species or event labels, case-sensitive, whitespace-ignored).  
- **Captioning** (bioacoustic description, scored with SPIDEr).  
- **Lifestage classification**, **call-type classification**, **individual counting**, **vocalization-type classification**.  
- Future: **behavior labeling**, **fine-grained detection with timestamps** (research questions, not v1).

Tasks explicitly **out of scope** in v1:

- General-purpose ASR (LibriSpeech, Common Voice, TEDLium, GigaSpeech).  
- Speech translation (CoVoST, FLEURS).  
- General audio scene QA (AudioCaps, Clotho-AQA, WavCaps) — except where the *same underlying dataset* appears in BEANS-Zero (ESC-50, for environmental sound classification).  
- Music understanding (MuChoMusic, MMAU music split).  
- Speech emotion / gender / accent recognition (IEMOCAP, VoxCeleb).  
- MCQ audio reasoning (MMAU audio/speech splits).  
- Singlish understanding (IMDA), code-switching (SEAME, ASCEND).

The framework could handle these tasks — nothing architectural prevents it — but shipping them as BEANS-Next registry content dilutes the bioacoustics-first positioning. If someone wants general-audio coverage, they can either use AudioBench / UltraEval-Audio (which exist and are maintained) or add their own non-bundled configs in user space.

### 1.3 Constraints

**Constraint 1 — No private-repo imports (with one optional exception).** The public library may not `import` anything from `esp_research`, `data_synth`, or any `esp_*` namespace. Every piece of functionality must be owned by the public package. Note: `earthspecies/beans-zero` is *public*, but we don't import it either (Constraint 5).

**Exception:** `esp_data` may be used as an **optional, guarded** dataset backend (no hard dependency; not listed in core dependencies). When selected via config, BEANS-Next may attempt `import esp_data` and use it to load BEANS-Zero rows more quickly; otherwise HuggingFace loaders remain the default. When `esp_data` is not installed, BEANS-Next must fail with a clear `ImportError` that tells the user how to switch back to HuggingFace.

**Constraint 2 — Scrubbable.** Nothing specific to your private research stack leaks into the public code: no internal dataset metadata, no private URLs, no research-model-weights paths beyond the public HF repos, no judge prompts that encode unreleased taste.

**Constraint 3 — Convergence-friendly.** Design for future convergence with `esp_research`, not immediate merger. Identical protocol signatures, identical registry key names, identical YAML schema keys, no gratuitous renames. (Unchanged from v2.5.)

**Constraint 4 — Minimal core, heavy work in sidecars.** The `beans_next` core package has no torch, no vllm, no transformers dependency. `pip install beans-next` is fast and small. Model-serving code lives in per-model **launcher scripts** under `examples/servers/`, each with its own venv. The core library only speaks HTTP.

**Constraint 5 — Clean break from `earthspecies/beans-zero`.** No `import beans_zero`. No dependency. No Callable compat shim. The two packages coexist but are architecturally independent. This keeps BEANS-Next free to evolve the HTTP contract and avoids legacy API debt.

### 1.4 Design principle

BEANS-Next is intentionally standalone for public release, but its core abstractions are shaped to remain interoperable with the broader `esp_research` evaluation framework through compatible config schemas, result objects, and adapter interfaces.

Design for **eventual convergence, not immediate merger**. Every abstraction choice is evaluated against both "does this make BEANS-Next clean today" and "does this keep the bridge into `esp_research` cheap later." When those pull in different directions, the tie goes to the public library — a released library with slightly-awkward reintegration is strictly better than a delayed library with perfect reintegration.

### 1.5 Design principle 2 — HTTP only

BEANS-Next talks to models **only** over HTTP. Every inference backend is someone else's server, started from a launcher script, never imported in-process.

This is what UltraEval-Audio arrived at in v1.1 after the "everyone's torch version conflicts" problem bit them. We start there. The core library is small, fast to install, and GPU-free. Launchers handle torch/vllm/transformers/naturelm/af3 complexity in isolated venvs. Consequences:

- No `[vllm]` / `[naturelm]` / `[af3]` / `[openai]` extras. `pip install beans-next` is the full install.  
- No in-process `VLLMClient` / `NatureLMAudioClient` / etc. Their equivalents are launcher scripts.  
- No "isolated runtime" IPC mechanism. Launchers are just separate processes with their own venvs, communicating over HTTP.  
- One inference adapter: `HttpClient`. Its request/response schema is the library's main public contract.

### 1.6 Non-negotiable contract

The following are architectural commitments, not implementation details. Changing any of them is a breaking change requiring a major version bump:

1. **The benchmark core only speaks `predictions_v1`.** No backend ever runs in-process; no Python import path substitutes for an HTTP call.  
2. **Launchers must implement `POST /predict`, `GET /info`, and `GET /health`.** All three are mandatory — `/info` is not optional (§4.3).  
3. **All official reproduction paths go through this contract.** Paper reviewers, downstream users, CI — nothing bypasses HTTP.  
4. **`InferenceModel` has one shipped implementation: `HttpClient`.** The protocol exists for `esp_research` convergence compatibility; additional implementations are not planned for v1 and should not be added speculatively.

Everything else in the document is negotiable. These four are not.

### 1.6.1 Aspirations (tracked, not release-blocking)

These are goals the project works toward and documents, but they do not gate v1 launch. They live here — separate from the non-negotiable contract — so the project can ship on a reasonable timeline without claiming validation work it hasn't done.

- **NatureLM-audio BEANS-Zero reproduction.** BEANS-Next should, eventually, reproduce published BEANS-Zero numbers for NatureLM-audio 1.0 and 1.1 side-by-side within documented tolerance. This is how the benchmark earns its credibility — if it can't reproduce the model it's built around, it's not trustworthy for other models. But producing those numbers requires sustained GPU access that the project author doesn't have at v1 launch. **v1 ships without this validation.** The first user with GPU access who runs `reproduce_all.sh configs/paper/beans_zero_naturelm_side_by_side.yaml` against public BEANS-Zero generates these numbers; we document what they produce in `REPRODUCING_THE_PAPER.md` post-launch. Until then, the library's claim is "pipeline works end-to-end on small samples; full-suite reproduction unverified."  
- **Tier-1 launcher CI coverage.** Ideally all Tier-1 launchers run against CI on every core release. At v1 launch this means `dummy` only, because GPU CI is a separate problem. `naturelm-v1.0` and `naturelm-v1.1` ship as code, conformance-tested locally once, not CI-gated.  
- **Cross-model baselines.** Running other audio-LMs (Qwen2-Audio, Qwen3-Audio, Audio Flamingo 3\) through BEANS-Next to produce comparison numbers. Planned; not a Phase 1 or Phase 2 blocker.

The discipline this section enforces: every time someone wants to add a release-blocking commitment that isn't strictly about the HTTP contract or library architecture, it goes here instead. Shipping dates don't slip for aspirations.

### 1.7 Two deliverables

This project is two things, not one:

**Deliverable A — the benchmark core (`beans_next` package).** Runner, datasets, prompts, post-processing, metrics, judges, artifacts, caches, CLI. Python package published to PyPI as `beans-next`. No torch in its dependency tree.

**Deliverable B — the serving kit (`examples/servers/` \+ `scripts/`).** The HTTP contract spec (§4), officially-maintained launchers (§5.2) including `naturelm-v1.0` and `naturelm-v1.1`, launcher docs, reproduction scripts, per-paper launcher `requirements.txt` pinning. Lives in the same repo for now; could split to a sibling repo later if launcher churn pollutes the core's release cadence.

Framing the project as two deliverables matters because under Path B the launchers are not illustrative examples — they are how anyone actually evaluates a model. A user who can install `beans_next` but can't start a launcher gets zero benchmark done. The serving kit earns equal design weight; its maintenance commitments (§5.2) are part of the public contract, not a convenience.

For BEANS-Next specifically, the `naturelm-v1.0` and `naturelm-v1.1` launchers are part of Deliverable B from day one — they are how the aspirational reproduction goal (§1.6.1) becomes achievable once someone with GPU access runs them at scale.

Release cadences are intentionally independent. A launcher requirements bump does not require a core release. A core release does not force launcher reruns. Tight coupling of the two is the failure mode this split exists to prevent.

### 1.8 Intentionally not in core

These are exclusions by design, not missing features:

- **No in-process model backends.** No `VLLMClient`, no `HFTransformersClient`, no `NatureLMClient`. All heavy inference runs behind HTTP.  
- **No extras groups for model dependencies.** `pip install beans-next` has no `[vllm]`, `[naturelm]`, `[openai]`. Model-specific dependencies live in launcher `requirements.txt` files.  
- **No direct OpenAI/Anthropic/Gemini clients.** Users wanting to benchmark API models run LiteLLM or the `openai_compatible_proxy` launcher and point `HttpClient` at it. The library does not vendor API SDKs.  
- **No general-audio task coverage.** No LibriSpeech, CoVoST, AudioCaps, MMAU, IEMOCAP, etc. in the bundled `registry/`. BEANS-Next is bioacoustics. Users wanting general-audio eval use AudioBench or UltraEval-Audio, which exist and are actively maintained.  
- **No `earthspecies/beans-zero` dependency or import.** BEANS-Next does not call `run_benchmark`, does not depend on `beans_zero` as a pip package, and does not provide Callable-adapter compat. The two packages share only the dataset on HuggingFace.  
- **No built-in server supervisor / process manager.** `reproduce.sh` and `reproduce_all.sh` do minimal lifecycle management (start, wait-for-health, kill). Anything more sophisticated (restart policies, GPU pool scheduling, multi-node) is out of scope — use `systemd`, `supervisord`, Kubernetes, or whatever the user already has.  
- **No requirement that launcher authors use Python.** The HTTP contract is the API. Rust, Go, Node.js, Bash-plus-curl launchers are all valid if they implement `predictions_v1` \+ `/info` \+ `/health`.  
- **No isolated-runtime IPC mechanism.** UltraEval-Audio v1.1 built this to dispatch different models with conflicting deps from one process. Path B sidesteps the problem by using separate processes from the start. No IPC, no subprocess management, no venv orchestration inside the library.  
- **No in-process model weight loading.** The benchmark never `torch.load`s anything. Launchers do.

If any of these become limitations in practice, we add them in a future major version with a clear justification. Defaulting to "not in core" until proven otherwise is the design stance.

---

## 2\. The user story, walked end-to-end

Everything in the rest of the document serves these three workflows. Read this section before anything else.

### 2.1 "Benchmark NatureLM-audio 1.0 and 1.1 on BEANS-Zero" (the headline story)

```shell
# Step 1: stand up NatureLM-audio 1.0 (public weights)
cd examples/servers/naturelm-v1.0
uv sync
./serve.sh --port 8000

# Step 1' (parallel terminal): stand up NatureLM-audio 1.1 (gated weights — needs HF_TOKEN)
cd examples/servers/naturelm-v1.1
uv sync
export HF_TOKEN=hf_...     # access to EarthSpeciesProject/naturelm-audio-1.1.00-private
./serve.sh --port 8001

# Step 2: run the benchmark against both side-by-side
pip install beans-next
beans-next run --config configs/paper/beans_zero_naturelm_side_by_side.yaml

# -> writes results/<run_id>/ with per-(model, dataset) metrics for both versions.
# The bundled config has both models (naturelm-v1.0 at :8000 and naturelm-v1.1 at :8001)
# and all BEANS-Zero tasks in one run.
```

Three terminals, five commands, one side-by-side report. If the user only has one model, they skip the second server and run `beans-next run` against a single-model config.

The `configs/paper/beans_zero_naturelm_side_by_side.yaml` ships with BEANS-Next. It encodes the paper's exact task selection, prompt templates, and post-processing so the numbers are reproducible by anyone.

### 2.2 "Benchmark my own bioacoustic model"

A researcher has a new bioacoustic audio-LM. They write a 50-line FastAPI launcher (or copy `examples/servers/naturelm-v1.0/serve.py` as a template) that exposes `POST /predict` \+ `GET /info` \+ `GET /health`. Then:

```shell
# Terminal 1: start their launcher
cd my-model-launcher
./serve.sh --port 8000

# Terminal 2: write a 5-line model YAML
pip install beans-next
cat > my-model.yaml <<EOF
my-model:
  adapter: http
  url: http://localhost:8000/predict
  audio_payload: base64_wav
  response_schema: predictions_v1
EOF

beans-next run --model my-model.yaml --suite beans_zero_core
# -> writes results/ with per-task accuracies, detection APs, captioning SPIDEr.
```

The user's model gets scored on the exact same BEANS-Zero tasks as NatureLM-audio, with the exact same post-processing and metric code. Comparable results, no forking.

### 2.3 "Reproduce our paper"

You publish a paper with BEANS-Next-based results. A reviewer wants to verify row 7 (NatureLM-audio 1.1 on `unseen-species-tax`):

```shell
git clone beans-next && cd beans-next
pip install -e .
export HF_TOKEN=hf_...
./scripts/reproduce.sh naturelm-v1.1 unseen-species-tax
# -> starts the naturelm-v1.1 launcher, waits for health, runs benchmark, shuts down server
```

Or the full paper grid:

```shell
export HF_TOKEN=hf_...
./scripts/reproduce_all.sh configs/paper/beans_next_paper_table_1.yaml
# -> iterates over every (model, task) row, starting each server once,
#    running all its rows, then moving to the next model.
```

Because the server stays warm across tasks within a model, `reproduce_all.sh` is substantially faster than running each row one-off. For the NatureLM-audio side-by-side report this matters a lot — reloading 8B params × 2 versions is the expensive step, and `reproduce_all.sh` amortizes it over all BEANS-Zero tasks.

### 2.4 "Just kick the tires"

For someone who wants to verify the pipeline works before installing any real model:

```shell
# one terminal
./examples/servers/dummy/serve.sh --port 8000

# another terminal
beans-next run --model examples/configs/dummy.yaml --task beans_zero_esc50 --limit 5
```

The dummy server echoes fixed strings. Thirty seconds, zero GPU, no HF token. New users verify their install and their understanding of the HTTP contract before downloading multi-gigabyte model weights.

---

## 3\. What exists today, what gets extracted, what gets left behind

The extraction is the load-bearing step. Under Path B, most of `data_synth.llms` **does not move into the public library** — the in-process inference clients become launcher scripts instead.

### 3.1 From `data_synth.llms` (the "model side")

| Component | Verdict | Notes |
| :---- | :---- | :---- |
| `LLM` protocol (`_base.py`) | **Lift, rename \+ alias** | Becomes `InferenceModel` in `beans_next.api.types`. Method signature tightened: `generate(requests: list[Request]) -> list[Response]`. `LLM = InferenceModel` as Tier 2 alias (§6.5). In practice the protocol has one implementation: `HttpClient`. |
| `messages_to_dicts` | **Lift as-is** | Pure utility, no private deps. Lives in `beans_next.api.messages`. |
| Audio-tag placeholders (`<Audio><AudioHere></Audio>`) | **Lift as-is** | Prompt-layer convention. Carried so launchers know how to align audio with text. |
| `prepare_multimodal_content` \+ audio I/O helpers | **Lift, rewrite I/O** | Replace `esp_data.io.read_audio` with `soundfile`. Used on client side (encoding base64 for HTTP) and documented for launchers (decoding). |
| `build_llm` factory | **Leave behind (as a factory)** | No factory needed when there's one adapter. Replaced by a direct `HttpClient(config)` constructor call. If we ever re-add backends, we'll re-add the factory. |
| `OpenAIClient` / `AnthropicClient` / `GeminiClient` | **Leave behind** | Users benchmarking API models run LiteLLM or a vLLM OpenAI-compatible proxy and point `HttpClient` at it. Library does not ship API-specific clients. |
| `VLLMClient` | **Leave behind → becomes `examples/servers/vllm/`** | A `vllm serve` wrapper plus a tiny FastAPI sidecar translating OpenAI-compatible output to our `predictions_v1` response schema. Alternatively the sidecar speaks OpenAI format directly and `HttpClient` has an `openai_compatible` mode. |
| `MLXLMClient` | **Leave behind** | If someone wants MLX, they write a launcher. |
| `NatureLMAudioClient` | **Leave behind → becomes `examples/servers/naturelm/`** | A FastAPI wrapper over the NatureLM-audio HF model class. Own venv, own `requirements.txt`. No special case for the library. |
| `AudioFlamingoClient` | **Leave behind → becomes `examples/servers/af3/`** | Same pattern. |
| `CustomLLM` base class | **Leave behind** | Not needed when there's no in-process extension path. |
| `RuleBasedLLM` \+ `rule_based/*` | **Leave behind** | Tied to synthesis; not benchmark-relevant. |
| `batch_complete` helper on vLLM | **Reference in launcher** | Launchers can support batched requests; `HttpClient` sends batched payloads; launchers decide how to exploit batching. |

**Net result:** almost none of `data_synth.llms` literal code ships. What ships is the **protocol shape** (`InferenceModel`), the **audio I/O helpers**, and the **prompt-tag convention**. The inference clients themselves all move to launcher scripts, where their finicky deps live with them.

### 3.2 From `esp_research` and `projects/beans-zero` (the "eval side")

Eval side is mostly unchanged from v2.3 — Path B doesn't affect it.

| Component | Verdict | Notes |
| :---- | :---- | :---- |
| `EvaluatesModelOnTasks` protocol | **Lift, rename \+ alias** | Canonical name: `Evaluator`. `EvaluatesModelOnTasks = Evaluator` as Tier 2 alias. |
| `EvalTaskConfig` / `TaskType` / `TargetDatasetSplit` | **Lift as-is** | `TaskType` enums for v1: `asr`, `translation`, `classification`, `detection`, `captioning`, `qa`, `mcq`. |
| `evals_registry` | **Lift, add alias** | Canonical: `TASK_REGISTRY`. `evals_registry` as Tier 2 alias. Has one entry in v1 (see §6.3). |
| `MetricConfig` / `Metric` / `MetricOutput` | **Lift as-is** | Pydantic wrappers. |
| `@register_scorer` \+ scorer function registry | **Lift as-is** | Keep name. |
| `spider` scorer | **Lift as-is** | Generalize out of `beans_zero/utils.py`. Document Java/SPICE requirement. |
| `BEANSZeroEvaluator` | **Lift skeleton, not wrapper** | Extract streaming/batching/post-processing into the generic runner. Drop BEANS-specific knobs. |
| `BEANSZeroEvalConfig` | **Leave behind** | Its knobs become CLI flags on the generic runner. |
| `PredictionPostProcessor` | **Lift, refactor into pipeline** | Split into `parsers.py` \+ `cleaners.py`; convention: parsers first. |
| `beans_zero_dataset_info.json` | **Leave behind** | Private dataset metadata. |
| `HttpClient` (`adapters/client.py`) | **Lift, formalize as the core public API** | This becomes *the* inference adapter. Explicit request schema versioning, retry policy, auth hooks, response validation, per-batch vs per-sample serialization, documented audio payload transport. See §4. |
| OpenAI FastAPI app | **Leave behind as-is; reference in launcher docs** | The existing FastAPI route logic is useful as a reference for `examples/servers/openai_compatible_proxy/`. |
| `LLMJudgeScorer` | **Lift, generalize, relocate** | Moves to `beans_next.judges.scorer`. Registry-driven Jinja templates. Under Path B it uses `HttpClient` to reach its judge model — consistent with everything else. |

**Missing from both, built fresh:**

- Dataset-loader dispatch (HF, HF streaming, JSONL, Polars) — internal, not a user extension point.  
- Prompt-template rendering (Jinja2 \+ audio tags) tuned for bioacoustic `instruction_text` conventions.  
- Bioacoustic scorers: `accuracy`, `f1`, `precision`, `recall`, `average_precision` (multi-label), `spider` (SPICE \+ CIDEr for captioning).  
- Optional bioacoustic judge template (for open-ended captioning-QA hybrids). Most BEANS-Zero scoring is deterministic — judges are a small secondary concern, not a headline feature.  
- Resume-from-breakpoint and two-layer cache (§7).  
- CLI (`beans-next run`, `list`, `describe`).  
- **Launcher scripts** (§5) including Tier-1 `naturelm-v1.0` and `naturelm-v1.1` — the anchor launchers.  
- `reproduce.sh` \+ `reproduce_all.sh` \+ `configs/paper/beans_zero_naturelm_side_by_side.yaml` (§8.2).  
- Documentation, examples, CI, PyPI packaging.

### 3.3 Public/private boundary audit

Same as v2.3. Must not ship: `esp_*`/`data_synth.*` imports, private URLs/paths, BEANS-Zero label metadata JSON, internal judge prompts, research weights (point at HF repos only), synthesis pipeline. Must audit: every client and loader for hardcoded defaults revealing internal setups. Repo hygiene: new git history, Apache-2.0, standard community files.

---

## 4\. The HTTP contract (the only public inference API)

Under Path B, everything about inference flows through `HttpClient`. Its request/response schemas are the library's single most important public API — more stable than any Python class, because every launcher on earth has to speak it.

### 4.1 Request schema (`predictions_v1`)

The client sends JSON to a `POST /predict` endpoint (name configurable). One request represents a batch of samples.

```json
{
  "schema_version": "predictions_v1",
  "requests": [
    {
      "sample_id": "beans_zero_esc50:0000",
      "messages": [
        {"role": "system", "content": "You are a bioacoustic audio classifier..."},
        {"role": "user", "content": "<Audio><AudioHere></Audio>\nWhat is the label for this environmental sound?"}
      ],
      "audio_inputs": [
        {
          "payload_type": "base64_wav",
          "data": "UklGRi...",
          "sample_rate": 16000
        }
      ],
      "generation_config": {
        "max_tokens": 256,
        "temperature": 0.0
      }
    }
  ]
}
```

`audio_inputs[i]` aligns with the i-th `<AudioHere>` placeholder in the rendered message. Supported `payload_type`s:

- `base64_wav` — inline base64-encoded WAV bytes. Simplest, works over any HTTP; bandwidth-heavy for long audio.  
- `file_path` — path the server can open directly. Only useful when client and server share a filesystem (same host or shared mount). Skips base64.  
- `file_url` — signed URL the server fetches. Useful for large-audio remote deployments.

A server declares which it supports in its `/info` endpoint (§4.3).

### 4.2 Response schema (`predictions_v1`)

```json
{
  "schema_version": "predictions_v1",
  "responses": [
    {
      "sample_id": "beans_zero_esc50:0000",
      "predictions": ["chainsaw"],
      "finish_reason": "stop",
      "usage": {"prompt_tokens": 120, "completion_tokens": 42},
      "latency_sec": 1.87,
      "error": null
    }
  ]
}
```

Any field except `sample_id` and `predictions` is optional. `predictions` is a list to allow n-best decoding; length 1 is the common case. `error` (nullable string) signals a sample-level failure — the benchmark runner records it but doesn't crash.

### 4.3 Server capability discovery (`GET /info`)

```json
{
  "name": "qwen3-audio-vllm-launcher",
  "model": "Qwen/Qwen3-Audio-7B-Instruct",
  "model_revision": "abc123...",
  "audio_payload_types": ["base64_wav", "file_path"],
  "max_batch_size": 8,
  "supports_batching": true,
  "schema_versions": ["predictions_v1"]
}
```

**`/info` is mandatory, not optional.** Every launcher — including minimal third-party launchers and experimental scripts — must implement it. The client calls `/info` once at startup; the runner aborts with a clear error if the endpoint is missing, malformed, or advertises a `schema_versions` list that excludes the one the client is using.

The fields aren't cosmetic. They serve four purposes:

1. **Reproducibility.** `name`, `model`, `model_revision` are recorded into `ModelPrediction.server_info` and surface in `RunSummary.model_identity`. Without this, two runs producing different numbers are undebuggable.  
2. **Cache invalidation.** `model_revision` is part of the inference cache key (§7.4). Upgrading the launcher's model weights invalidates stale cache entries automatically.  
3. **Schema negotiation.** `schema_versions` lets the client fail fast when talking to a launcher that speaks a newer or older contract.  
4. **Batching hints.** `max_batch_size` and `supports_batching` drive how the runner chunks work, avoiding 413 errors and letting the scheduler exploit server-side batching where available.

"My launcher is too simple to need `/info`" is not a valid reason to omit it — the dummy launcher (§5.4) is the minimum reference implementation and it implements `/info` in about 15 lines.

### 4.4 Error model

`HttpClient` distinguishes three error categories:

- **Transient** (timeout, 5xx, connection reset): retry with exponential backoff per the retry policy declared in the model YAML.  
- **Sample-level** (server returns a response with `error` populated): record in `ModelPrediction.error`, don't retry, don't abort the run.  
- **Fatal** (4xx except 408/429, schema validation failure, auth failure): abort the run with a clear error message.

Default retry policy: 3 attempts, exponential backoff from 1s to 30s, jitter. Overridable per-model.

### 4.5 Auth hooks

A model YAML can declare auth in one of three forms:

```
# Bearer token from env
auth:
  type: bearer
  token_env: MY_ENDPOINT_TOKEN

# Custom header
auth:
  type: header
  header: X-API-Key
  value_env: MY_API_KEY

# None (local launchers)
auth:
  type: none
```

mTLS support deferred to v2 unless someone needs it.

### 4.6 Batching and error semantics

The wire format is batch-based. The behavioral contract is specified below; launcher authors must follow these rules, and the client relies on them.

**Sample identity.**

- Every `sample_id` in a batch must be unique. Clients must validate this before sending; servers may reject duplicates with 400\.  
- Response-to-request matching is by `sample_id`, not by array position. Servers may return responses in any order; the runner pairs them by ID.  
- Servers must return exactly one response per request sample. Missing `sample_id`s in the response array are treated as fatal errors.  
- Extra `sample_id`s in the response (not in the request) are treated as fatal errors.

**Batch size.**

- Clients discover `max_batch_size` from `/info` and chunk their batches accordingly.  
- Servers that receive a batch exceeding `max_batch_size` must return HTTP 413 with an error body naming the limit.  
- Servers may internally split a batch of size N into multiple smaller sub-batches for scheduling, but must return exactly N responses in one HTTP response body. Streaming responses are out of scope for v1.

**Partial failure.**

- Sample-level failures are per-response-item via the `error` field. The batch call itself still returns HTTP 200\.  
- A batch HTTP call returns non-200 only for batch-scope failures: auth, schema validation, server unavailable, timeout. Sample-level failures never bubble up as HTTP errors.  
- The client records per-sample `error` strings in `ModelPrediction.error` and continues the run. The runner treats sample-level errors as failed predictions for scoring purposes (counted in `n_errors` in `DatasetResult`).

**Timeouts.**

- Timeouts are per-HTTP-call, not per-sample. The client's read timeout applies to the whole batch response.  
- Servers are responsible for chunking long batches internally so the total response fits under the client's timeout. The `/info` endpoint may advertise a recommended `max_batch_size` tuned for this.  
- If a client timeout fires mid-batch, all samples in that batch are lost — no partial results. This is a retry scenario, not a partial-failure scenario. The retry policy applies.

**Retry interaction.**

- Transient HTTP failures (timeout, 5xx, connection reset) trigger a retry of the whole batch per the retry policy.  
- Retries must be idempotent from the server's perspective — sending the same batch twice must produce equivalent results (modulo sampling-induced non-determinism).  
- Servers should deduplicate identical consecutive batch calls within a short window if the cost matters, but this is not required by the contract.

**Order guarantees.**

- The client does not assume request order within a batch is preserved on the server side.  
- The client does not assume response order matches request order — it matches by `sample_id`.  
- Across batches, the client preserves submission order for result-writing, so `predictions.jsonl` is deterministic given a fixed dataset order.

These rules exist because under-specifying batching is how launcher authors and client authors silently diverge. The golden rule: **`sample_id` is the only identity that matters, and `error` is the only way to signal partial failure.**

---

## 5\. Launcher scripts (`examples/servers/`)

Launchers are the thing the user actually touches when bringing a model. Each is independent, has its own venv, and implements the HTTP contract in §4. Under the two-deliverables framing (§1.5), launchers are *equal-weight deliverables* alongside the core library — not illustrative examples.

### 5.1 Layout

```
examples/servers/
├── README.md                       # "pick the right launcher for your model"
├── dummy/                          # onboarding: echoes fixed strings, zero deps
│   ├── serve.sh
│   ├── serve.py
│   └── requirements.txt            # fastapi, uvicorn
├── naturelm-v1.0/                  # NatureLM-audio 1.0 (public weights)
│   ├── serve.py
│   ├── requirements.txt            # pinned to earthspecies/NatureLM-audio SHA
│   └── README.md                   # downloads EarthSpeciesProject/NatureLM-audio
├── naturelm-v1.1/                  # NatureLM-audio 1.1 (gated weights; requires HF_TOKEN)
│   ├── serve.py
│   ├── requirements.txt            # pinned to earthspecies/NatureLM-audio SHA
│   └── README.md                   # documents gated-access request workflow
├── vllm/                           # general purpose: any HF chat model vLLM supports
│   ├── serve.sh                    # wraps `vllm serve` + FastAPI adapter sidecar
│   ├── adapter.py                  # OpenAI-compat -> predictions_v1 translation
│   ├── requirements.txt            # vllm==X.Y.Z, fastapi, uvicorn (pinned)
│   └── README.md
├── hf_transformers/                # generic HF model not served by vLLM
│   ├── serve.py
│   ├── requirements.txt
│   └── README.md
├── af3/                            # Audio Flamingo 3
│   ├── serve.py
│   ├── requirements.txt
│   └── README.md
└── openai_compatible_proxy/        # reference: wrap an existing OpenAI-format endpoint
    ├── serve.py
    ├── requirements.txt
    └── README.md
```

### 5.2 Support tiers

Not all launchers carry equal weight. Under the two-deliverables framing (§1.7), the serving kit makes hard maintenance commitments on some launchers and only best-effort on others. For BEANS-Next specifically, the NatureLM-audio launchers sit in Tier 1 because they are how the §1.6.1 reproduction aspiration is ever satisfied.

**Officially maintained (Tier 1 of the serving kit):**

- **`dummy`** — minimum reference, zero GPU, CI-tested. Canonical `/info` implementation reference.  
- **`naturelm-v1.0`** — BEANS-Next's anchor launcher. Wraps the public `EarthSpeciesProject/NatureLM-audio` HF model. Pinned `requirements.txt` matching a specific `earthspecies/NatureLM-audio` repo SHA. Regressions here block a release.  
- **`naturelm-v1.1`** — BEANS-Next's second anchor launcher. Wraps the gated `EarthSpeciesProject/naturelm-audio-1.1.00-private` HF model. Reads `HF_TOKEN` from env. Users without 1.1 access still get a working v1.0 launcher, so the gating is not a blanket blocker — it just degrades the side-by-side workflow to single-model.  
- **`openai_compatible_proxy`** — wraps any OpenAI-format endpoint (LiteLLM, hosted API, local vLLM OpenAI server). CI-runnable without GPU.  
- **`vllm`** — covers Qwen2-Audio, Qwen3-Audio, Gemma-Audio, Phi-4-multimodal, any HF chat model vLLM supports. Used to benchmark other audio-LMs on BEANS-Zero as baselines.

These five are:

- Version-pinned per release of the serving kit.  
- Code-maintained with regression fixes.  
- Covered by the conformance test on every core release (the GPU-free conformance check — schema, `/info`, batching semantics). Full-inference CI on the GPU-dependent ones (`naturelm-*`, `vllm`) is aspirational (§1.6.1).  
- Documented in `REPRODUCING_THE_PAPER.md` with expected runtime and GPU class.

**Reference implementations (Tier 2 of the serving kit):**

- **`hf_transformers`** — generic fallback for audio-LMs vLLM doesn't support.  
- **`af3`** — Audio Flamingo 3\.

These are:

- Shipped as working starting points, with pinned `requirements.txt` at the time of their last commit.  
- Not covered by CI on every core release.  
- Best-effort maintenance — regressions may land and may wait on external contribution to fix.  
- Expected to work at the tagged release; expected to drift over time as upstream model code changes.

**Why this split.** The NatureLM-audio launchers are Tier 1 because BEANS-Next's reason to exist is benchmarking bioacoustic audio-LMs, and the two NatureLM versions are the canonical models in that category today. Without them in Tier 1, the library is a framework with no reference content. The `vllm` and `openai_compatible_proxy` launchers are Tier 1 because they enable benchmarking any other audio-LM as a baseline, with minimal BEANS-Next-specific maintenance burden — they wrap upstream infrastructure.

The Tier 2 launchers track upstream APIs that change outside our control (Audio Flamingo, arbitrary HF chat models with idiosyncratic processors). Committing the public library to their maintenance would slow down core releases for reasons unrelated to BEANS-Next's architecture. Better to say honestly: here are reference implementations; here's how to fix them if they break for your model.

User implications:

- If you're benchmarking **NatureLM-audio 1.0** → use `naturelm-v1.0`, fully supported, no token required.  
- If you're benchmarking **NatureLM-audio 1.1** → use `naturelm-v1.1`, fully supported, requires `HF_TOKEN` with access to the gated HF repo.  
- If you're benchmarking **Qwen3 / Gemma / Phi audio variants** → use `vllm`, fully supported.  
- If you're benchmarking **a hosted API model** (GPT-4o-audio, Gemini) → use `openai_compatible_proxy` with LiteLLM, fully supported.  
- If you're benchmarking **AF3 or an odd HF model** → use the Tier 2 reference launcher, expect to possibly edit it.  
- If you're benchmarking **your own custom model** → copy `naturelm-v1.0/` as a starting template (it's a clean FastAPI \+ HF `from_pretrained` pattern).

### 5.3 Contract for a launcher

A launcher is a valid launcher if:

1. Running it exposes `POST /predict` accepting `predictions_v1` requests and returning `predictions_v1` responses.  
2. Running it exposes `GET /info` returning the capability discovery document (mandatory — §4.3).  
3. Running it exposes `GET /health` returning 200 when ready.  
4. It respects the batching and error semantics in §4.6 (sample\_id matching, partial failure via `error` field, 413 on max\_batch\_size exceeded).  
5. Its `requirements.txt` is pinned to specific versions (so paper reproduction is bit-exact).  
6. Its `README.md` documents: model family supported, GPU/RAM requirements, start command, common failure modes.

That's the entire contract. No Python class to subclass. No decorator to register. No import path. Launchers are just Unix processes with a documented HTTP API.

### 5.4 The `dummy` launcher

Ships with the core repo, no GPU, base `[fastapi, uvicorn]` deps only. Its `predict` endpoint returns a fixed response per `task_type`: "hello world" for ASR, "A" for MCQ, "label\_0" for classification. Users run it to verify the pipeline in \~30 seconds. The benchmark scores against this give garbage numbers — that's the point; you're verifying the plumbing, not the model.

### 5.5 Writing a new launcher

Someone with a model not in the bundled list writes a \~50-line FastAPI app. They can copy `hf_transformers/` as a starting point. The only real work is:

1. Load their model.  
2. In `POST /predict`, decode base64 audio → tensor, run inference, return text.  
3. Implement `/info` and `/health`.  
4. Follow the batching contract (§4.6) — sample\_id matching, per-item errors, 413 above max\_batch\_size.

If it runs, the benchmark can drive it. No library PR required.

---

## 6\. Public library architecture

### 6.1 High-level flow

```
flowchart LR
  cli[beans-next CLI] --> runner[BenchmarkRunner]
  runner --> reg[YAML loader<br/>resolves registry names]
  runner --> ds[DatasetLoader<br/>HF / HF-stream / JSONL / Polars]
  ds -.produces.-> de[DatasetExample]
  de --> prompt[PromptRenderer<br/>Jinja2 + audio tags]
  prompt -.produces.-> req[ModelRequest]
  req --> http[HttpClient<br/>ONLY adapter]
  http -.HTTP.-> ext((External<br/>launcher))
  ext -.HTTP.-> http
  http -.produces.-> pred[ModelPrediction]
  pred --> icache{{Inference<br/>cache}}
  pred --> pp[PostProcess pipeline<br/>parsers then cleaners]
  pp --> scorer[Deterministic scorers<br/>wer/bleu/accuracy/AP/spider]
  pp --> judge[Judges<br/>separate registry]
  scorer -.produces.-> sp[ScoredPrediction]
  judge -.produces.-> sp
  sp --> scache{{Scoring<br/>cache}}
  sp --> store[ResultStore]
  store -.produces.-> rs[RunSummary]
```

The one visible change from v2.3's diagram: `HttpClient` is the only model node, and it terminates at an external boundary (the launcher). Nothing heavy is in-process.

### 6.2 Package layout

```
beans_next/
├── __init__.py                    # re-exports: BenchmarkRunner, HttpClient, Request, Response, ...
├── cli.py
├── api/
│   ├── types.py                   # Request, Response, DatasetExample, ModelPrediction, ... (§7)
│   ├── messages.py                # messages_to_dicts, audio tag constants
│   └── http_schemas.py            # predictions_v1 request/response Pydantic models
├── config/
│   ├── loader.py                  # YAML loader, registry resolution, Shape A -> Shape B
│   ├── schemas.py                 # EvalTaskConfig, DatasetConfig, ModelConfig, ...
│   └── registries.py              # @register_* decorators, internal dispatch
├── models/
│   ├── base.py                    # InferenceModel protocol (one implementation only)
│   └── http.py                    # HttpClient — the entire model side of the library
├── datasets/
│   ├── base.py
│   ├── hf.py
│   ├── hf_streaming.py
│   ├── jsonl.py
│   └── polars.py
├── prompts/
│   ├── renderer.py                # Jinja2 + audio-tag handling; DatasetExample -> ModelRequest
│   └── audio_tags.py
├── post_process/
│   ├── pipeline.py
│   ├── parsers.py                 # extract_option_letter, parse_labels_comma, json_field, ...
│   └── cleaners.py                # strip_eos, normalize_whitespace, fuzzy_match_to_labels, ...
├── metrics/                       # DETERMINISTIC only — judges live separately
│   ├── base.py                    # MetricConfig, Metric, MetricOutput, @register_scorer
│   ├── classification.py          # accuracy, f1, recall, precision
│   ├── detection.py               # average_precision (multi-label)
│   └── captioning.py              # spider (SPICE + CIDEr)
├── judges/                        # optional LLM-as-judge scorer — own registry, own artifact file
│   ├── base.py                    # JudgeConfig, JudgeOutput, @register_judge
│   ├── scorer.py                  # JudgeScorer; uses HttpClient for the judge model too
│   └── templates/                 # Jinja: bioacoustic_open_qa_v1 (Phase 2; Phase 1 has no judges)
├── runner/
│   ├── runner.py                  # BenchmarkRunner
│   ├── batching.py                # streaming-aware batching
│   ├── checkpoint.py              # sample_id hashing, resume scan
│   ├── parallel.py                # worker pool (batched HTTP concurrency)
│   └── health.py                  # pre-run /health + /info probe of the launcher
├── results/
│   ├── store.py                   # all JSONLs + summary.json + checkpoint.json
│   └── schema.py
├── tasks/
│   ├── __init__.py                # TASK_REGISTRY (alias evals_registry), Evaluator protocol
│   └── audio_to_text.py           # the generic evaluator
├── registry/                      # BUNDLED YAMLs
│   ├── dataset/                   # beans_zero (single entry covering all 20+ subsets)
│   ├── prompt/                    # classification_default, detection_default, captioning_default, taxonomic_default, lifestage_default, ...
│   ├── model/                     # endpoint presets: naturelm_v1_0_local_8000, naturelm_v1_1_local_8001, dummy_local_8000
│   ├── judge/                     # bioacoustic_open_qa_v1 (Phase 2 only)
│   ├── eval_task/                 # one per BEANS-Zero subset: beans_zero_esc50, beans_zero_watkins, beans_zero_cbi, ...
│   ├── suite/                     # beans_zero_core (all BEANS-Zero subsets), beans_next_extensions (future)
│   └── pricing/                   # per-model token prices (optional)
├── examples/
│   ├── servers/                   # THE launcher directory — see §5
│   └── configs/                   # dummy.yaml, naturelm_v1_0_local.yaml, naturelm_v1_1_local.yaml
├── scripts/
│   ├── reproduce.sh               # one-model, one-task: server up, run, shut down
│   └── reproduce_all.sh           # iterate a paper table, keep server warm across tasks
├── configs/
│   └── paper/                     # YOUR PAPER'S EXACT CONFIGS (see §8.5)
└── tests/
```

### 6.3 Registries — smaller than v2.3

Path B and the "just use HttpClient" constraint together kill most of v2.3's registries. What survives:

**Internal dispatch registries (users never touch):**

- `MODEL_REGISTRY` — has one entry: `http`. Kept for the `adapter:` YAML key to dispatch, and for future-proofing. Not a user extension point.  
- `SCORER_REGISTRY` — internal dispatch for the shipped metrics.  
- `JUDGE_REGISTRY` — internal dispatch for the shipped judge scorer types.  
- `DATASET_LOADER_REGISTRY` — internal dispatch for HF / JSONL / Polars loaders.  
- `POSTPROCESS_STEP_REGISTRY` — internal dispatch for the short-name steps referenced in YAML.

**`InferenceModel` protocol — one shipped implementation.** `HttpClient` is it. The protocol exists for `esp_research` convergence compatibility (§6.5, Tier 2 aliases). Implementers working on the library should not add speculative `InferenceModel` subclasses — a second implementation means a new backend, which under Path B means a launcher, not a Python class. Adding one would reverse the architectural commitment in §1.4.

**Data registries (directories of YAMLs that ship with the library):**

- `registry/dataset/`, `registry/prompt/`, `registry/judge/`, `registry/eval_task/`, `registry/suite/`, `registry/pricing/` — all serve their v2.3 purpose unchanged.  
- **`registry/model/` — now functions as an "endpoint preset" directory.** Under Path B, a model YAML is not really a model config — it's an HTTP endpoint config: URL, auth, payload mode, retry policy, optional `max_batch_size` hint. The directory holds reusable presets: `registry/model/local_vllm_8000.yaml` (pointing at the default vLLM launcher port), `registry/model/paper_qwen3_audio.yaml` (the exact preset used for a paper row), and so on. The directory name stays `model/` for backward compatibility with `esp_research`\-shaped configs; its semantic role is endpoint presets. Where the prose says "model YAML" throughout the doc, read "endpoint preset."

**What went away from v2.3:**

- `TASK_REGISTRY` as a real registry — has one entry, so it's just a constant. `evals_registry` remains as a Tier 2 alias returning a trivial dict-like for compat.  
- `@register_model` as a user-facing extension point — users don't add backends; they add launchers.  
- The framing of all five code registries as "public extension points." They're internal implementation details; the user surface is YAML.

### 6.4 Run configs, post-process, judges, schemas, caching

These are all unchanged from v2.3. Rather than repeat, here are the pointers:

- **Run configs**: two shapes (registered suites \+ inline), task/execution split. Your BEANS-Zero YAML still works unchanged.  
- **Polars loader \+ `split` overloading**: supported; `split:` means HF-split for HF loaders, subset identifier for Polars loaders, documented per-loader.  
- **Parser-first post-process convention**: one YAML block, two internal modules (`parsers.py` \+ `cleaners.py`), convention documented.  
- **Judge/metric separation**: `metrics:` and `judges:` as separate YAML blocks, separate registries, separate artifact files (`judge_outputs.jsonl`).  
- **Canonical schemas**: §7 below.  
- **Run artifacts \+ two-layer cache**: §7 below.

### 6.5 Tiered API stability

- **Tier 1 — Frozen public API (semver-guarded):**  
  - CLI verbs: `run`, `list`, `describe`, `score-from-file`.  
  - **HTTP contract**: `predictions_v1` request/response schemas, `/info`, `/health`. This is the biggest public commitment in the library — every launcher on earth depends on it.  
  - Top-level imports: `BenchmarkRunner`, `HttpClient`, `InferenceModel`, and all five canonical schemas.  
  - YAML schema keys (same as v2.3 §4.2).  
  - Result artifact file names (§7).  
- **Tier 2 — Internal compatibility aliases:**  
  - `LLM = InferenceModel`, `register_llm = register_model` (a no-op in v1), `LLM_REGISTRY` alias, `evals_registry` alias, `EvaluatesModelOnTasks = Evaluator`.  
- **Tier 3 — Internal implementation:**  
  - Module layout under `runner/`, `post_process/`, registry internals.

---

## 7\. Canonical data schemas and run artifacts

Unchanged from v2.3 §6 \+ §7 except as noted.

### 7.1 The five Pydantic schemas

`DatasetExample`, `ModelRequest`, `ModelPrediction`, `ScoredPrediction`, `RunSummary`. Each carries `sample_id` and `schema_version: Literal[1]`. Full definitions in v2.3 §6 (all carry over).

**One change in v2.4:** `ModelPrediction` gains an optional `server_info: dict[str, Any]` field populated from the launcher's `/info` response at run start. This lets the run artifact record the exact launcher and its `model_revision` for reproducibility. `ModelIdentity` derives from `server_info` when present.

### 7.2 Pipeline

```
DatasetExample ──render──▶ ModelRequest ──HTTP──▶ ModelPrediction ──parse+postprocess+score──▶ ScoredPrediction
                                                                                                       │
                                                                                aggregate ◀────────────┘
                                                                                       │
                                                                                       ▼
                                                                                  RunSummary
```

### 7.3 Artifact layout (one run)

```
<output_dir>/<model>/<run_id>/
├── run_config.yaml              # resolved config (post-registry)
├── model_identity.json          # ModelIdentity, including server /info snapshot
├── predictions.jsonl            # one ModelPrediction per sample
├── processed_predictions.jsonl  # one ScoredPrediction per sample, post-parser + post-cleaner, pre-metric
├── scored_predictions.jsonl     # same rows, post-metric
├── judge_outputs.jsonl          # only if judges ran
├── summary.json                 # RunSummary — top-level result
└── checkpoint.json              # resume state
```

### 7.4 Two-layer cache

- **Inference cache** (SQLite by default): keyed by `(model_identity_hash, prompt_version, sample_id, audio_fingerprint, generation_config_hash)`. Value: `ModelPrediction`. Hit skips the HTTP call.  
- **Scoring cache**: keyed by `(prediction_artifact_id, parser_version, postprocess_version, scorer_config_hash)`. Value: `ScoredPrediction`. Hit skips parse/post-process/score without re-inference.

This is especially valuable under Path B because it lets someone iterate on post-processing rules without spinning the launcher server back up.

### 7.5 Reproducibility guarantee

Every `summary.json` records: `library_version`, `code_git_sha`, `run_config_hash`, `prompt_version`, `postprocess_version`, per-metric `scorer_version`, `model_identity` (including the launcher's `model_revision`), `seed`. Two runs with identical values in these fields produce bit-exact results for deterministic scorers and within-float-noise for stochastic ones. This is the contract Phase 2's exit criterion checks.

---

## 8\. Build-out plan

Path B significantly compresses Phases 1 and 3\. Phase 2 grows slightly because multi-launcher support replaces the in-process backends story. A new Phase 2.5 runs parallel to Phase 2 and covers the paper-reproduction workflow.

### 8.1 Phase 0 — Clean-room extraction (3–4 days)

1. Create `beans-next` repo (new git history, Apache-2.0, skeleton pyproject, no extras).  
2. Copy the protocol / schemas / audio I/O helpers from `data_synth.llms` — *not* the inference clients.  
3. Copy `esp_research.metrics`, `esp_research.evals.base`, registry helpers, `HttpClient`, `spider` scorer.  
4. Extract the streaming/batching/post-processing logic from `BEANSZeroEvaluator` into the generic `BenchmarkRunner`.  
5. Extract `LLMJudgeScorer` into `beans_next.judges`.  
6. Run `grep -r "esp_\|data_synth" beans_next/` — must return zero.  
7. `pip install -e .` succeeds with **no torch, no vllm, no transformers** in dependency tree.

**Exit criterion:** fresh venv, `pip install -e .`, `python -c "import beans_next"` works. Install size under 100 MB.

### 8.2 Phase 1 — Pipeline end-to-end on small samples (1–2 weeks)

Phase 1's job is to build the benchmark pipeline and validate it end-to-end on **small samples** — not full-suite GPU runs. The constraint: the project author does not have sustained GPU access. Large-scale reproduction (§1.6.1 aspiration) happens post-launch when a collaborator runs `reproduce_all.sh`.

What Phase 1 must prove: if you give BEANS-Next a working HTTP launcher and a BEANS-Zero task, it produces valid artifacts, computes the right metrics, and doesn't silently corrupt anything. What Phase 1 does **not** prove: that NatureLM-audio 1.0 or 1.1 actually reproduces published numbers. That proof is deferred.

**Core library track:**

1. Add canonical schemas (§7) and config schemas.  
2. Wire `BenchmarkRunner` end-to-end. `HttpClient`, HF \+ HF-streaming dataset loaders, prompt renderer, post-process pipeline, scorer, result store.  
3. Ship these registered items:  
   - **Datasets:** `beans_zero` (one dataset registry entry covering all BEANS-Zero subsets — esc50, cbi, watkins, humbugdb, lifestage, call-type, unseen-species-{cmn,sci,tax}, unseen-genus-*, unseen-family-*, zf-indiv, enabirds, hiceas, rfcx, gibbons, dcase, captioning).  
   - **Scorers:** `accuracy`, `f1`, `precision`, `recall`, `average_precision`, `spider`.  
   - **Prompts:** bioacoustic prompt templates aligned with BEANS-Zero's `instruction_text` conventions (one per task\_type).  
   - **Post-process steps:** `parse_labels_comma` (case-sensitive, whitespace-ignored per BEANS-Zero rules), `fuzzy_match_to_labels`, `strip_eos`, `normalize_whitespace`.  
   - **Eval tasks \+ suite:** `registry/suite/beans_zero_core.yaml` encoding the full BEANS-Zero subsets with correct task\_types and metrics per subset.  
4. CLI: `beans-next run`, `list`, `describe`.  
5. Ship the **`dummy` launcher** (`examples/servers/dummy/`) — Tier 1; returns fixed strings per task\_type; zero GPU; canonical `/info` reference implementation.  
6. Ship a smoke test: `scripts/smoke_test.sh` that runs the dummy launcher \+ benchmark on 5 samples across at least three task\_types (classification, detection, captioning).  
7. Ship the **contract conformance test** (`scripts/check_launcher.sh <url>`).

**Serving kit track (launcher code, not launcher runs):**

8. **Write** the `naturelm-v1.0` launcher (`examples/servers/naturelm-v1.0/`). Wraps `EarthSpeciesProject/NatureLM-audio` via `NatureLM.from_pretrained(...)` \+ `NatureLM.infer.Pipeline`. Pinned `requirements.txt`. Audio windowing advertised in `/info`. Tier 1\. *Written and committed; actual GPU test run is separate — see "Release validation" below.*  
9. **Write** the `naturelm-v1.1` launcher (`examples/servers/naturelm-v1.1/`). Same structure as v1.0; points at gated `EarthSpeciesProject/naturelm-audio-1.1.00-private`; reads `HF_TOKEN`; fails with a clear message when absent or lacking access. Tier 1\.  
10. Ship `configs/paper/beans_zero_naturelm_side_by_side.yaml` — lists both launchers (port 8000 \+ 8001), full BEANS-Zero suite, paper-aligned prompts and post-processing.  
11. Ship `scripts/reproduce.sh` and `scripts/reproduce_all.sh` with server-lifecycle management (`uv sync`, start, health-check, run, kill).

**Phase 1 exit criteria (all must pass, none require a multi-hour GPU run):**

- **(a) Pipeline smoke test.** `pip install beans-next` → start dummy launcher → `beans-next run` on 5 BEANS-Zero samples → valid result artifacts. Under 30 seconds. No GPU.  
- **(b) Dummy launcher conformance.** The dummy launcher passes `scripts/check_launcher.sh` — schema version, `/info`, batching, per-sample error handling per §4.6.  
- **(c) `naturelm-v1.0` launcher small-sample run.** Somewhere with a GPU (a short cloud GPU session, a teammate's workstation, a Colab notebook), the `naturelm-v1.0` launcher runs successfully against **5–20 samples per task\_type** from BEANS-Zero. Exit criterion is "outputs are well-formed, metrics compute without error, no crashes," not "numbers match the paper."  
- **(d) `naturelm-v1.1` launcher small-sample run \+ auth.** Same as (c) with `HF_TOKEN` set. Additional check: the launcher surfaces a clear, actionable error message when `HF_TOKEN` is absent or doesn't grant access.  
- **(e) Side-by-side report on small samples.** The full side-by-side config runs end-to-end on 5–20 samples per dataset with both launchers, produces a single `RunSummary` with both models' metrics, and a prose `summary.md` renders them in a side-by-side table. Numbers are not required to match any reference — the goal is "the pipeline produces a coherent side-by-side artifact."

If any of (a)–(e) fail, Phase 1 is not done. Phase 2 does not start.

**Release validation (pre-launch, post-Phase-2, one-time):**

Before the public launch tag, someone with sustained GPU access runs:

```shell
export HF_TOKEN=hf_...
./scripts/reproduce_all.sh configs/paper/beans_zero_naturelm_side_by_side.yaml
```

against the full BEANS-Zero suite on both models. They record the metrics in `REPRODUCING_THE_PAPER.md` along with GPU type, runtime, library SHA, and launcher requirements SHA. If those numbers match published BEANS-Zero results within reasonable tolerance, the aspiration in §1.6.1 is satisfied at launch. If they don't, the mismatch is documented honestly and the launch claim is softened to "pipeline is correct; reproduction has N% drift from paper, under investigation."

This step is **not Phase 1's responsibility** and does not block writing v1 code. It's a separate pre-launch checklist item owned by whoever on the team can book GPU hours.

### 8.3 Phase 2 — Additional launchers \+ polish (1–2 weeks)

With Phase 1 complete, BEANS-Next has a pipeline that works end-to-end on small samples and launcher code ready for anyone with GPU access to run at scale. Phase 2 widens the launcher ecosystem and polishes the user experience. This phase is also written without requiring author GPU access — launcher code is written, small-sample-tested where possible (cloud GPU sessions are fine), and validated by collaborators at scale.

**Serving kit — additional Tier 1:**

1. Ship the **`openai_compatible_proxy` launcher** — wraps any OpenAI-format endpoint. Covers "benchmark a hosted model via LiteLLM" use cases. CI-runnable without GPU. Tier 1\.  
2. Ship the **`vllm` launcher** (`examples/servers/vllm/`) — general wrapper around `vllm serve` \+ `predictions_v1` adapter sidecar. Used when someone wants to benchmark any HF chat audio-LM (Qwen2-Audio, Qwen3-Audio, Gemma-Audio, Phi-4-multimodal) on BEANS-Zero as a baseline. Tier 1 in terms of code maintenance; GPU validation deferred like the naturelm launchers.

**Serving kit — Tier 2 reference launchers:**

3. Ship the **`hf_transformers` launcher** — generic fallback for audio-LMs vLLM doesn't support. Best-effort.  
4. Ship the **`af3` launcher** — Audio Flamingo 3\. Best-effort.

All new launchers must pass the conformance test (which is GPU-free — it targets `/info` and schema validation, not full inference correctness).

**Core library polish:**

5. Add `JudgeScorer` under `beans_next.judges` \+ one public bioacoustic judge template (for open-ended bioacoustic captioning-QA hybrids). Fewer judges than v2.5 proposed because bioacoustic scoring is mostly deterministic.  
6. Add `PolarsDataset` for users who have BEANS-Next-compatible data in Parquet.  
7. Resume-from-breakpoint (§7.3), `--workers` parallelism.  
8. Add additional BEANS-Zero-adjacent datasets if/when ESP publishes them.

**Exit criteria:**

- All six launchers pass the conformance test (GPU-free).  
- At least one non-NatureLM audio-LM launcher (e.g., `vllm` \+ Qwen2-Audio) runs end-to-end on 5–20 samples somewhere. Numbers are baselines, not targets.  
- Documentation site with "bring your model" tutorial, "write a launcher" tutorial, "reproduce the NatureLM-audio paper" tutorial.

**Core library polish:**

5. Add `JudgeScorer` under `beans_next.judges` \+ one public bioacoustic judge template (for open-ended bioacoustic captioning-QA hybrids). Fewer judges than v2.5 proposed because bioacoustic scoring is mostly deterministic.  
6. Add `PolarsDataset` for users who have BEANS-Next-compatible data in Parquet.  
7. Resume-from-breakpoint (§7.3), `--workers` parallelism.  
8. Add additional BEANS-Zero-adjacent datasets if/when they're published by ESP: any extensions to BEANS-Zero, new bioacoustic benchmarks from the community.

**Exit criteria:**

- All six launchers pass conformance.  
- At least one non-NatureLM audio-LM (e.g., Qwen2-Audio via the `vllm` launcher) can be scored end-to-end on the BEANS-Zero suite. Numbers are baselines, not targets.  
- Documentation site with "bring your model" tutorial, "write a launcher" tutorial, "reproduce the NatureLM-audio paper" tutorial.

### 8.4 Phase 3 — Caching, cost, public launch (1 week)

1. Ship the two-layer cache (§7.4).  
2. Cost and token accounting (read from `Usage` in `ModelPrediction`, aggregate into `RunSummary`).  
3. Generalized retry policies in `HttpClient`.  
4. `score-from-file` command.  
5. Per-model prompt overrides in eval\_task YAML.  
6. Docs site (mkdocs) with the three tutorials. "Reproduce the NatureLM-audio paper" documents the current state honestly: pipeline works, full-suite numbers await the post-launch validation run.  
7. Write `REPRODUCING_THE_PAPER.md` with the launch-day honest version: which SHA reviewers should check out, which launcher versions, `HF_TOKEN` instructions, expected runtime on reference GPU, known nondeterminism sources. At launch this document says "validation run pending"; post-launch it gets a PR adding the actual numbers from the first collaborator run.

**Exit criterion:** public launch. PyPI release. The launch claim is honest: "BEANS-Next is a bioacoustics benchmark pipeline with Tier-1 NatureLM-audio 1.0/1.1 launcher support. Pipeline correctness validated on small samples. Full-suite reproduction of published BEANS-Zero numbers is documented post-launch (§1.6.1)."

### 8.5 Phase 4 — Reintegration preparation (background)

Unchanged from v2.3. Build the `beans_nextmark_esp_bridge` package when the public library is stable, or do the absorb-path if `esp_research` ever opens up.

---

## 9\. Risks and mitigations

**Risk: two-terminal pattern confuses first-time users.** Mitigation: the dummy launcher \+ a prominent "30-second smoke test" in the README. `reproduce.sh` collapses the pattern to one command for the paper workflow.

**Risk: HTTP transport bandwidth for long audio.** Mitigation: `file_path` and `file_url` payload types for same-host or remote-storage deployments. Base64 is the default because universal; users scale up by switching payload type without changing adapters.

**Risk: launcher maintenance burden.** Mitigation: each launcher is minimal (\<200 lines typically). Pinned requirements mean they don't silently break when upstream models change. The two-tier split (§5.2) bounds the commitment explicitly — `dummy`, `naturelm-v1.0`, `naturelm-v1.1`, `vllm`, `openai_compatible_proxy` get code-maintenance and regression-fix support; `hf_transformers` and `af3` are reference implementations that may drift. Tier 2 launchers that go permanently stale can be removed in minor releases without affecting Tier 1 or the core library.

**Risk: launcher tiers confuse users who assume full support for everything under `examples/servers/`.** Mitigation: the README in `examples/servers/` leads with the tier table. Each Tier 2 launcher's own README states its maintenance status in the first paragraph. The CI badge on the repo reflects Tier 1 launchers only; Tier 2 has a separate "last-known-good" badge pinned to a commit hash.

**Risk: NatureLM-audio 1.1 gated weights block users.** A significant fraction of BEANS-Next's value is the side-by-side 1.0 vs 1.1 comparison, but 1.1 is gated. Users without access can still run 1.0, which covers the published-numbers case, but they cannot reproduce 1.1 results. Mitigation: (a) `naturelm-v1.0` requires no token and runs everything the paper reports against 1.0 on public BEANS-Zero. (b) `naturelm-v1.1`'s README prominently documents the HF gated-access request flow (request URL, expected response time, who to email if it stalls). (c) The side-by-side config automatically degrades to single-model mode if `naturelm-v1.1`'s `/health` is unreachable, so users without access don't hit a cryptic failure — the run completes with 1.0 results and a documented skip for 1.1.

**Risk: BEANS-Next ships without real NatureLM-audio reproduction numbers at launch.** Phase 1 explicitly exits on small-sample pipeline correctness, not full-suite reproduction (§8.2). If no teammate with GPU access runs `reproduce_all.sh` before launch, the launch claim is reduced to "pipeline works, reproduction unverified." Mitigation: the §1.6.1 aspiration is named honestly in the doc; the README states plainly that full-suite reproduction is post-launch; and the first post-launch release (v1.1 of BEANS-Next, not of NatureLM-audio — naming collision aside) is blocked on someone committing the full `reproduce_all.sh` numbers to `REPRODUCING_THE_PAPER.md`. Worst case, we ship honest and update later; we never ship a claim we haven't validated.

**Risk: silent scoring differences after extraction.** Same as v2.3: bit-exact snapshot of a 20-sample run on the private side before extraction; diff must be clean after.

**Risk: HTTP contract locks in too early.** It's the biggest semver commitment in the library. Mitigation: `schema_version: "predictions_v1"` in every payload, launchers declare supported versions in `/info`, loader handles multiple versions for at least one minor release. A `predictions_v2` can ship without breaking existing launchers.

**Risk: the library becomes "just HttpClient" and users ask why they need it at all.** Real. The value prop is: HttpClient \+ dataset/prompt/post-process/score/report/resume/cache pipeline \+ bundled benchmarks \+ paper-reproduction workflow. The inference hop is small; everything around it is the point.

**Risk: someone wanting to benchmark an API model (GPT-4o-audio) finds it awkward.** Two options: (a) run LiteLLM as a local proxy, point `HttpClient` at it — documented. (b) Use the `openai_compatible_proxy` launcher which is exactly this pattern prepackaged. Common case is one config file, not a custom launcher.

**Risk: paper reviewers fail to reproduce results because of launcher dep drift.** Mitigation: pinned launcher `requirements.txt` committed with each paper's `configs/paper/<paper_id>/`. Launcher version frozen to a git SHA referenced in the paper. Reviewers check out that SHA.

**Risk: divergence from `esp_research` makes merge-back impossible.** Same as v2.3: quarterly merge review flagging drift.

---

## 10\. Open questions

1. **License.** Apache-2.0 recommended, matching `earthspecies/beans-zero` and `earthspecies/NatureLM-audio`.  
2. **Who owns Phase 0 extraction?** One person should do the clean-room extraction from `data_synth.llms` \+ `esp_research` to keep naming consistent.  
3. **Gated-weights access workflow robustness.** The `naturelm-v1.1` launcher depends on HF gated-access for `EarthSpeciesProject/naturelm-audio-1.1.00-private`. Open questions: How long does access typically take to grant? Is there a risk the gating policy changes post-launch? Should the launcher include a "verify access" subcommand that calls the HF API to confirm the token works *before* attempting to download weights, so users get fast feedback? Recommend: yes, ship a `serve.sh --check-access` flag that exits cleanly if the token doesn't work.  
4. **Launcher repo split timing.** Start in the same repo (`examples/servers/`). Revisit if the two-deliverable release cadence starts pulling launchers and core in incompatible directions, or if Tier 2 launchers accumulate enough that they dominate the repo. Sibling repo name if split: `beans-next-servers`.  
5. **HTTP contract schema ownership.** `predictions_v1` is a public API. It should live in a documented spec file (`docs/http_contract.md`) in addition to the Pydantic code, so launcher authors in other languages (Rust, Go) can implement it without reading Python.  
6. **Who runs the first full `reproduce_all.sh` post-launch, and when?** Per §1.6.1 \+ §8.2 "Release validation," a teammate with GPU access runs the full BEANS-Zero reproduction on both NatureLM-audio versions at some point after launch. Open question: is there a named person and a target date for this? The release claim softens until it happens. A reasonable target: within 4 weeks of launch, documented in `REPRODUCING_THE_PAPER.md` as the first published reproduction. If the numbers drift from the paper, the drift itself is a publishable finding.  
7. **Naming collision with BEANS-Zero.** "BEANS-Next" is an explicit extension/successor to BEANS-Zero, but the existing `earthspecies/beans-zero` pip package stays around per §1.1. Is there any near-term plan to sunset `beans-zero` the package once BEANS-Next stabilizes, or do they coexist indefinitely? If indefinitely, the project needs a one-page "which one should I use" decision tree in the public docs.

---

## 11\. One-line takeaway

**BEANS-Next is a bioacoustics audio-LM benchmark library from Earth Species Project — successor to BEANS-Zero — built around a non-negotiable HTTP contract, shipped as two deliverables (core \+ serving kit), with Tier-1 `naturelm-v1.0` and `naturelm-v1.1` launchers as the anchor models; v1 launches when the pipeline runs end-to-end on small samples (no GPU required from the author), and full-suite reproduction of NatureLM-audio on BEANS-Zero is an aspiration (§1.6.1) validated post-launch by a collaborator with GPU access.**  