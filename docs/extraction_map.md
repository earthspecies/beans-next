# BEANS-Next — source extraction map (Increment 0)

This document records the **clean-room** mapping from ESP research sources into `beans_next/`, per `AGENT_SPEC.md`, `IMPLEMENTATION_PLAN.md`, `INCREMENTS.md`, and `DESIGN.md`. It is descriptive: actual porting happens in later increments; paths here are **read-only references** to upstream trees.

---

## Model-side repository

| Item | Detail |
|------|--------|
| Worktree path | `~/code/esp-research__data-synth-runsv3/projects/NatureLM-audio-data-synth` |
| Expected branch | `data-synth/runsv3` |
| Expected module | `data_synth/llms/` |
| Status | Available via git worktree checkout (preferred for multi-agent work). |

### Working assumptions (while keeping BEANS-Next clean-room)

These follow `AGENT_SPEC.md` §5.1 / §SOURCE CODE EXTRACTION PLAN and `DESIGN.md` §3.1; **file names may differ** once the tree is opened.

- **Lift (rewrite in BEANS-Next terms):** LLM-style protocol → `InferenceModel` in `beans_next` (HTTP-shaped, not in-process clients).
- **Lift:** `messages_to_dicts` (or equivalent) → `beans_next/api/messages.py`.
- **Lift:** audio placeholder convention (e.g. `<Audio><AudioHere></Audio>`) → `beans_next/prompts/audio_tags.py` (and prompt/renderer docs).
- **Rewrite:** multimodal / audio I/O helpers that today depend on private stack (e.g. `esp_data` readers) → use **`soundfile`** (and related public deps) in core; no `data_synth` imports remain.
- **Do not port:** OpenAI / Anthropic / Gemini clients, VLLM / NatureLM / MLX / AF3 / rule-based / `build_llm` factory patterns — those become **`examples/servers/*` launchers**, not `beans_next` imports.

---

## Eval-side repository (inspected on disk)

| Item | Detail |
|------|--------|
| Worktree paths | `/home/marius_miron_earthspecies_org/code/esp-research__openai-adaptater/` and `/home/marius_miron_earthspecies_org/code/esp-research__beans-zero-project/` |
| Branch `paul/openai-adaptater` | Present as a **local** git ref (`67554abb5444b33823dc6f82339d0f4db0dae26b`). |
| Branch `paul/beans-zero-project` | Available via worktree (`4811315` at time of writing). |

---

## Mapping table (source → `beans_next`)

| Source repo / path | Source branch | Source module / file | Destination in `beans_next` | Action | Notes / rationale |
|--------------------|---------------|----------------------|----------------------------|--------|-------------------|
| `~/code/esp-research__data-synth-runsv3/projects/NatureLM-audio-data-synth` | `data-synth/runsv3` | `data_synth/llms/` (protocol / base module; exact filename TBD) | `beans_next/models/base.py` | **Rewritten** | BEANS-Next defines HTTP-first `InferenceModel`; do not retain `data_synth` imports (`AGENT_SPEC.md` §5.1). |
| `~/code/esp-research__data-synth-runsv3/projects/NatureLM-audio-data-synth` | `data-synth/runsv3` | `data_synth/llms/` (message helpers) | `beans_next/api/messages.py` | **Rewritten** / partial **copy** | Pure utilities may be near–verbatim; still re-homed and audited for forbidden imports. |
| `~/code/esp-research__data-synth-runsv3/projects/NatureLM-audio-data-synth` | `data-synth/runsv3` | `data_synth/llms/` (audio placeholder / multimodal text conventions) | `beans_next/prompts/audio_tags.py` | **Rewritten** | Align placeholders with HTTP `predictions_v1` + prompt renderer (`DESIGN.md` §3.1, §4.1). |
| `~/code/esp-research__data-synth-runsv3/projects/NatureLM-audio-data-synth` | `data-synth/runsv3` | `data_synth/llms/` (audio encode / decode helpers tied to `esp_data` or similar) | `beans_next/api/messages.py` or small helper module under `beans_next/api/` | **Rewritten** | Replace private I/O with **`soundfile`**-based paths (`AGENT_SPEC.md` §5.1). |
| `~/code/esp-research__data-synth-runsv3/projects/NatureLM-audio-data-synth` | `data-synth/runsv3` | Any inference client modules (OpenAI, vLLM, NatureLM, etc.) | *(none in core)* → `examples/servers/…` (later increments) | **Dropped** from core | Clients are launchers only; core stays HTTP-only (`DESIGN.md` §1.5–1.8). |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/adapters/client.py` | `paul/openai-adaptater` (see note) | `HttpClient` | `beans_next/models/http.py` | **Rewritten** | Upstream implements legacy payload shape + `httpx`; BEANS-Next must implement **`predictions_v1`**, `/info`, `/health`, batching semantics (`AGENT_SPEC.md` §8, `DESIGN.md` §4). No `esp_research` imports. |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/adapters/client_config.py` | same | `HttpClientConfig`, `HttpAuthConfig` | `beans_next/config/` (YAML-aligned types) and/or `beans_next/models/http.py` | **Rewritten** | Fold into BEANS-Next model endpoint config; drop `CLIConfig` coupling from `esp_research.configs`. |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/adapters/audio_utils.py` | same | `coerce_audio_bytes`, `encode_audio`, WAV helpers | `beans_next/api/` or `beans_next/models/http.py` (as private helpers) | **Rewritten** | Useful logic reimplemented against **`predictions_v1`** audio fields; avoid numpy/torch assumptions where spec requires a lightweight core (`INCREMENTS.md` global rules). |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/protocols/model.py` | same | `InferenceModel` (ESP: broad `__call__`) | `beans_next/models/base.py` | **Dropped** / **replaced** | ESP `InferenceModel` is not the BEANS-Next contract; BEANS-Next defines a new protocol aligned with HTTP batching. |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/evals/base.py` | same | `EvalTaskConfig`, `TaskType`, `TargetDatasetSplit`, `evals_registry` | `beans_next/config/` · `beans_next/tasks/` (registry) | **Rewritten** | Decouple from `esp_data.DatasetConfig`; keep **schema-compatible** discriminated configs where possible (`DESIGN.md` §1.3, §6.5). |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/evals/base.py` | same | `validate_has_expected_model_output` | `beans_next/runner/` or `beans_next/tasks/` | **Rewritten** / optional | May map to launcher conformance + schema checks instead of evaluator class hooks. |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/protocols/eval.py` | same | `EvaluatesModelOnTasks` | `beans_next/tasks/` (evaluator protocol) | **Rewritten** | Renamed / aliased per `DESIGN.md` §3.2 (`Evaluator`). |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/metrics/base.py` | same | `MetricConfig`, `Metric`, `MetricOutput` | `beans_next/metrics/base.py` | **Rewritten** | Same concepts; new package path; no `esp_research` imports. |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/metrics/score_functions.py` | same | `@register_scorer`, `get_scorer`, concrete scorers | `beans_next/metrics/*.py` | **Rewritten** | Registry pattern preserved; implementations trimmed to bioacoustics-relevant set (`AGENT_SPEC.md` §10). |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/metrics/utils.py` | same | metric utilities | `beans_next/metrics/` | **Rewritten** | As needed by lifted scorers. |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/metrics/external/spice/` and `.../cider/` | same | SPICE / CIDEr / Java bridge helpers | `beans_next/metrics/captioning.py` (or subpackage) | **Rewritten** / **copied** (vendor subset) | `spider` depends on SPICE+CIDEr (`DESIGN.md` §3.2); keep licensing / third-party notes when copying Stanford jars or scripts. |
| `/home/marius_miron_earthspecies_org/code/esp-research/src/esp_research/utils/registry.py` | same | `Registry` | `beans_next/registry/` internals or `beans_next/config/registries.py` | **Rewritten** | Same idea; must not import `esp_research`. |
| `/home/marius_miron_earthspecies_org/code/esp-research/projects/beans-zero/src/beans_zero/evaluator.py` | `paul/beans-zero-project` (spec) / current tree | `BEANSZeroEvaluator` | `beans_next/runner/runner.py`, `beans_next/runner/batching.py`, `beans_next/post_process/*` | **Rewritten** (split) | **Do not** ship monolith (`AGENT_SPEC.md` §5.2); depends on `torch`, `esp_data` today — not acceptable in core. |
| `/home/marius_miron_earthspecies_org/code/esp-research/projects/beans-zero/src/beans_zero/evaluator.py` | same | `BEANSZeroEvalConfig`, `BEANSZeroEvalResult` | `beans_next/config/` + `beans_next/results/schema.py` | **Rewritten** | Knobs move to CLI + YAML (`DESIGN.md` §3.2). |
| `/home/marius_miron_earthspecies_org/code/esp-research/projects/beans-zero/src/beans_zero/utils.py` | same | `spider` scorer registration | `beans_next/metrics/captioning.py` | **Rewritten** | Register under BEANS-Next scorer registry. |
| `/home/marius_miron_earthspecies_org/code/esp-research/projects/beans-zero/src/beans_zero/configs/beans_zero/beans_zero_dataset_info.json` | same | dataset metadata | — | **Dropped** | Private / non-portable metadata (`AGENT_SPEC.md` §5.2, `DESIGN.md` §3.3). Use HuggingFace `EarthSpeciesProject/BEANS-Zero` and public registry YAML. |
| `/home/marius_miron_earthspecies_org/code/esp-research/projects/beans-zero/src/beans_zero/configs/beans_zero/beans_zero_eval_cfg.yml` | same | eval YAML | `beans_next/registry/**` (later) | **Rewritten** | Paths and knobs normalized to BEANS-Next registry layout (`DESIGN.md` §6.2). |
| `/home/marius_miron_earthspecies_org/code/esp-research/projects/beans-zero/openai_eval_cfg.yml`, `openai_model_cfg.yml` | same | OpenAI-specific configs | — | **Dropped** | Private / environment-specific presets; patterns may inform **launcher** docs only (`DESIGN.md` §3.2 OpenAI FastAPI note). |
| `/home/marius_miron_earthspecies_org/code/esp-research/projects/beans-zero/src/beans_zero/cli.py` | same | CLI entry | `beans_next/cli.py` (later increment) | **Rewritten** | New `beans-next` CLI verbs (`IMPLEMENTATION_PLAN.md` §1.2). |

**Branch note:** Rows referencing `esp_research/` use the file layout visible under `/home/marius_miron_earthspecies_org/code/esp-research/` at the time of mapping. For strict per-branch parity, check out `paul/openai-adaptater` and `paul/beans-zero-project` and diff against this table.

---

## Increment 0 destination files (skeleton scope)

These are the **first** BEANS-Next modules named in `INCREMENTS.md` for I0-B / I0-C; extraction above feeds them:

| Destination | Primary upstream themes |
|-------------|-------------------------|
| `beans_next/api/types.py` | Canonical run schemas (new; influenced by evaluator dataclasses / metric outputs, not copied wholesale). |
| `beans_next/api/http_schemas.py` | `predictions_v1` wire format (new; supersedes legacy adapter JSON). |
| `beans_next/api/messages.py` | `data_synth` message dict utilities (pending repo) + any small helpers split from HTTP client. |
| `beans_next/models/base.py` | `InferenceModel` protocol (BEANS-Next definition). |
| `beans_next/models/http.py` | `esp_research.adapters.client` concepts, rewritten to contract. |
| `beans_next/prompts/audio_tags.py` | `data_synth` audio tag constants (pending repo). |
| `beans_next/results/schema.py` | Optional; mirrors artifact / summary shapes (`AGENT_SPEC.md` §12). |

---

## Clean-room constraints

No `beans_next/` source file may import or embed forbidden **package paths** or substrings from the private stack.

### Forbidden substrings (non-exhaustive; use scans below)

- `esp_research`
- `esp_data`
- `data_synth`
- `beans_zero`
- Broad pattern covering ESP-prefixed internal packages: `esp_` (matches `esp_research`, `esp_data`, and other `esp_*` namespaces called out in specs)

### Required verification commands

Per `AGENT_SPEC.md` §5.3 / §SOURCE CODE EXTRACTION PLAN:

```bash
grep -r "esp_\|data_synth" beans_next/
```

**Expected:** zero lines of output (no matches).

Broader scan (recommended; aligns with `DESIGN.md` §1.3 and `INCREMENTS.md` global rules):

```bash
grep -rE 'esp_research|esp_data|data_synth|beans_zero' beans_next/
```

Optional (faster, if `ripgrep` is installed):

```bash
rg 'esp_research|esp_data|data_synth|beans_zero|\besp_' beans_next/
```

**Install / import check** (`IMPLEMENTATION_PLAN.md` §0.4):

```bash
uv run python -c "import beans_next"
```

---

## What’s intentionally **not** ported

- **Inference clients** (OpenAI, Anthropic, Gemini, vLLM, NatureLM in-process, MLX, Audio Flamingo, custom `build_llm` factories) — launcher-only (`AGENT_SPEC.md` §5.1, `DESIGN.md` §1.8, §3.1).
- **`BEANSZeroEvaluator` as a single class** — split into runner, batching, post-process, scoring (`AGENT_SPEC.md` §5.2, `DESIGN.md` §3.2).
- **Dataset metadata JSON** (e.g. `beans_zero_dataset_info.json`) and other **private paths / credentials / machine-local YAML** (`AGENT_SPEC.md` §5.2).
- **Callable / in-process model execution** in core — HTTP only (`AGENT_SPEC.md` §2).
- **Dependency on** `earthspecies/beans-zero` **the package** — no `beans_zero` pip / import path in `beans_next` (`DESIGN.md` §1.1, §1.3).
- **ESP `InferenceModel` protocol** (`esp_research.protocols.model`) — replaced by BEANS-Next HTTP protocol (`DESIGN.md` §1.6).

---

## Open questions / follow-ups

1. **Obtain** `NatureLM-audio-data-synth` @ `data-synth/runsv3` and reconcile this table with **actual** filenames under `data_synth/llms/`.
2. **Fetch** `paul/beans-zero-project` in the eval clone and note any file moves vs `paul/openai-adaptater`.
3. Confirm whether **numpy** in audio helpers is acceptable in core for I0, or whether bytes-only + `soundfile` suffices for iteration 1.
4. Decide where **SPICE / CIDEr Java** assets live in-repo (vendor copy vs optional extra) when `spider` lands in I3.
