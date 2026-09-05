# Apocrita benchmark client

This launcher runs dataset loading, HTTP requests, and scoring on an Apocrita
CPU node. Run the model server as a separate GPU Slurm job and provide its
`/predict` URL.

One-time setup after copying the repository to scratch:

```bash
mkdir -p /gpfs/scratch/acw777/beans-next-runs/logs
```

Start with one example from the tiny suite:

```bash
cd /gpfs/scratch/acw777/beans-next-code
BEANS_NEXT_PREDICT_URL=http://MODEL_NODE:PORT/predict \
BEANS_NEXT_LIMIT=1 \
sbatch examples/apocrita/run_benchmark.sbatch
```

Then remove `BEANS_NEXT_LIMIT` and select the intended suite, for example:

```bash
BEANS_NEXT_PREDICT_URL=http://MODEL_NODE:PORT/predict \
BEANS_NEXT_SUITE=beans_next_all_tiers_hf \
sbatch examples/apocrita/run_benchmark.sbatch
```

The defaults use the downloaded snapshot at
`/gpfs/scratch/acw777/esp/beans-next`, keep caches and environments in GPFS,
write results below `/gpfs/scratch/acw777/beans-next-runs/results`, and resume
an interrupted run. Set `BEANS_NEXT_SHARED_FILESYSTEM=0` for a model endpoint
outside Apocrita so audio is sent as base64 instead of GPFS paths.

## Text-only LLM evaluation

Create the vLLM runtime in a CPU job:

```bash
mkdir -p /gpfs/scratch/acw777/beans-next-runs/logs
sbatch examples/apocrita/setup_text_eval_runtime.sbatch
```

After that job succeeds, download the public Qwen snapshots in another CPU job:

```bash
sbatch examples/apocrita/download_text_models.sbatch
```

Read the generated model manifest under `/gpfs/scratch/acw777/beans-next-models`.
Submit `run_text_eval.sbatch` with an exact snapshot path and revision. Use
Andrena for the 7B models and SAE A100 80GB for Qwen3-30B-A3B. The default
pilot runs 16 examples from every task at concurrency 16, 32, and 64. A full
run evaluates the informed condition and a deterministic 10 percent silent
condition while the model remains loaded.

For a startup smoke test, set `BEANS_NEXT_PILOT_LIMIT=1` and
`BEANS_NEXT_CONCURRENCY_GRID=16` at submission time.

## NVIDIA Audex 30B-A3B (BEANS-Next tiers 1-3)

Audex (`nvidia/Nemotron-Labs-Audex-30B-A3B`, the 30B MoE — not the 2B
variant) needs its own runtime and model cache, separate from the text-eval
vLLM venv above: it requires a vendor vLLM architecture plugin
(`inference_scripts_vllm/audioqa_scripts/` inside the model repo, package
`audex_30b_a3b_vllm`) editable-installed, and plain `vllm serve` will not
find the architecture without it. See `examples/servers/audex/README.md` for
the full serving contract (chat template, codec-token suppression, sampling
arms).

1. Build the base Audex vLLM runtime (CPU job):

   ```bash
   sbatch examples/apocrita/setup_audex_runtime.sbatch
   ```

2. Download the Audex snapshot (CPU job; skips the audio-generation and
   text-only checkpoint variants, ~67 GB instead of 71.7 GB):

   ```bash
   sbatch examples/apocrita/download_audex.sbatch
   ```

   Read the resulting manifest under `/gpfs/scratch/acw777/beans-next-models`
   for the resolved `snapshot_path`.

3. Re-run setup with the snapshot path to editable-install the plugin into
   the same runtime venv:

   ```bash
   BEANS_NEXT_AUDEX_SNAPSHOT_PATH=<snapshot_path> \
     sbatch examples/apocrita/setup_audex_runtime.sbatch
   ```

4. Serve + evaluate on a 2-GPU 80 GB node (SAE `pilot_sae_gpu`, or
   `gpushort` for a ≤1h smoke test):

   ```bash
   BEANS_NEXT_AUDEX_MODEL_PATH=<snapshot_path>/checkpoint_folder_full \
   BEANS_NEXT_AUDEX_MODEL_REVISION=<resolved_revision> \
   BEANS_NEXT_RUN_MODE=smoke \
     sbatch --partition gpushort --constraint "ampere&80G" --gres=gpu:2 \
       --time=00:55:00 examples/apocrita/run_audex_eval.sbatch
   ```

   `BEANS_NEXT_AUDEX_MODEL_PATH` must point at `checkpoint_folder_full`
   inside the snapshot, not the snapshot root: the root `config.json` is only
   a repo-level manifest (no `architectures`/`auto_map`), so `vllm serve`
   fails to resolve the model class if pointed there directly.

   `BEANS_NEXT_RUN_MODE` is `smoke` (2 rows, tier 1 only), `pilot` (32 rows
   per task across tiers 1-3), `full` (all 51k+ rows, 48h with
   `--resume-from` support), or `beans_zero` (all 91,965 rows of
   `beans_zero_core`, esp_data backend). `BEANS_NEXT_AUDEX_DECODING_ARM`
   (`greedy` or `nvidia_sampling`) selects the decoding config for the pilot
   A/B — both run through the same launcher via `AUDEX_EXTRA_BODY_JSON`, no
   code change. Tier 4 (multi-audio) and BirdSet are out of scope: the vLLM
   server is started with `--limit-mm-per-prompt '{"audio": 1}'`.

5. For `beans_zero_core` (BEANS-Zero), first stage the 16 kHz esp_data
   subtree to local scratch (public bucket, no auth needed):

   ```bash
   sbatch examples/apocrita/stage_beans_zero_esp_data.sbatch
   ```

   Then run/chain evaluation with `BEANS_NEXT_RUN_MODE=beans_zero`,
   `BEANS_NEXT_DATA_BACKEND=esp_data`, `BEANS_NEXT_SUITE=beans_zero_core`.
   Per-subset clip caps (`beans_next/registry/beans_zero_max_duration_seconds.json`)
   are honoured automatically by the Audex launcher (head-truncated, never
   padded — see `examples/servers/audex/serve.py`), matching the official
   protocol and NatureLM. `examples/apocrita/chain_audex_beans_zero.sh`
   chains ≤55min `gpushort` jobs with `--resume-from` until
   `suite/beans_zero_core/suite_summary.json` appears, mirroring
   `chain_audex_full.sh`'s pattern for `beans_next_tiers_1_3_hf`.

## Google Gemma 4 12B (encoder-free Unified; BEANS-Zero + BEANS-Next tiers 1-3)

`google/gemma-4-12B-it` (11.95B dense params, not gated, Apache 2.0) is
encoder-free: raw 16 kHz audio is sliced into 40 ms frames and projected
straight into the LM embedding space, so it needs vLLM **≥ 0.23.0** with the
`audio` extra rather than the vendor plugin Audex needs. It fits one 80 GB
(or even 40 GB) GPU, so this gets its own runtime, model cache, and sbatch
scripts, separate from the Audex runtime above. Both BEANS-Zero and
BEANS-Next tiers 1-3 data are already staged on scratch from the Audex run —
there is no staging job to run first.

1. Download the snapshot (CPU job, `computeshort`, ~24 GB; not gated, no
   checkpoint variants to skip):

   ```bash
   sbatch examples/apocrita/download_gemma4.sbatch
   ```

   Read the resulting manifest under `/gpfs/scratch/acw777/beans-next-models`
   for the resolved `snapshot_path` and `resolved_revision` — pin both into
   any `run_gemma4_eval.sbatch` submission and into the two chain scripts
   below (`GEMMA4_CHAIN_SNAPSHOT`/`GEMMA4_CHAIN_MODEL_REVISION` and
   `GEMMA4_TIERS_CHAIN_SNAPSHOT`/`GEMMA4_TIERS_CHAIN_MODEL_REVISION`), the
   same way the Audex chain scripts pin
   `4e7e342045736382ddf3e2952c313847a08642b8`.

2. Build the Gemma 4 vLLM runtime (CPU job) at
   `$SCRATCH/beans-next-gemma4-runtime` — this does **not** touch
   `beans-next-audex-runtime` or the shared text-eval runtime:

   ```bash
   sbatch examples/apocrita/setup_gemma4_runtime.sbatch
   # then, once the download manifest above exists:
   BEANS_NEXT_GEMMA4_SNAPSHOT_PATH=<snapshot_path> \
     sbatch examples/apocrita/setup_gemma4_runtime.sbatch
   ```

   The second pass is a hard gate, not a formality: it asserts
   `Gemma4UnifiedForConditionalGeneration` is present in the installed
   vLLM's architecture registry, and that `AutoProcessor` for the downloaded
   snapshot exposes a `feature_extractor` (its absence is the documented
   cause of silently-ignored audio — a fluent, well-formed, entirely
   meaningless prediction column with no error anywhere). Both fail loudly.
   If the stable `vllm[audio]` wheel turns out to lack encoder-free Unified
   support, set `BEANS_NEXT_VLLM_VERSION`/`BEANS_NEXT_VLLM_WHEEL_URL` to a
   pinned/nightly wheel and re-run pass 1 — this is meant to be discovered
   here, in a CPU job, not on a GPU.

3. **Before any full run**, gate the serving backend with the Decision-1
   audio-sensitivity probe (`gpushort`, 1 GPU, ~30 min):

   ```bash
   BEANS_NEXT_GEMMA4_MODEL_PATH=<snapshot_path> \
   BEANS_NEXT_GEMMA4_MODEL_REVISION=<resolved_revision> \
   BEANS_NEXT_RUN_MODE=probe \
     sbatch --partition gpushort --constraint "ampere&80G" --gres=gpu:1 \
       --time=00:55:00 examples/apocrita/run_gemma4_eval.sbatch
   ```

   This starts the launcher (vLLM by default) and runs
   `examples/apocrita/gemma4_audio_probe.py`: 12 rows, 3 conditions each
   (real clip / digital silence / a different clip with a different
   ground-truth label), plus a best-effort cross-check against a direct HF
   Transformers forward pass on 3 rows. It exits non-zero if the three
   conditions return identical text for most rows — meaning audio is not
   reaching the model. **Do not skip this because `/health` is green.** If it
   fails, set `BEANS_NEXT_GEMMA4_BACKEND=transformers` (in-process HF
   Transformers serving, no vLLM server; ~3-5x slower, no continuous
   batching) for every run after it and re-probe.

4. Serve + evaluate on a single-GPU 80 GB node (SAE, or `gpushort` for a ≤1h
   smoke test):

   ```bash
   BEANS_NEXT_GEMMA4_MODEL_PATH=<snapshot_path> \
   BEANS_NEXT_GEMMA4_MODEL_REVISION=<resolved_revision> \
   BEANS_NEXT_RUN_MODE=smoke \
     sbatch --partition gpushort --constraint "ampere&80G" --gres=gpu:1 \
       --time=00:55:00 examples/apocrita/run_gemma4_eval.sbatch
   ```

   `BEANS_NEXT_RUN_MODE` is `probe` (Decision-1 gate, see above), `smoke` (2
   rows, tier 1 only), `pilot` (32 rows per task across tiers 1-3, or
   `beans_zero_smoke` with `BEANS_NEXT_DATA_BACKEND=esp_data`), `full` (all
   51k+ tiers rows, chained with `--resume-from`), or `beans_zero` (all
   91,965 rows of `beans_zero_core`, esp_data backend).
   `BEANS_NEXT_GEMMA4_DECODING_ARM` (`greedy`, the reported arm, or
   `vendor_sampling`) selects the decoding config, again via
   `GEMMA4_EXTRA_BODY_JSON` with no launcher code change. The vLLM server
   (when `BEANS_NEXT_GEMMA4_BACKEND=vllm`) starts with
   `--limit-mm-per-prompt '{"image": 4, "audio": 1}'` and
   `--reasoning-parser gemma4`; tier 4 (multi-audio) and BirdSet are out of
   scope, the same boundary as Audex.

5. Chain the two full runs with `examples/apocrita/chain_gemma4_beans_zero.sh`
   and `examples/apocrita/chain_gemma4_tiers.sh` (both need
   `GEMMA4_CHAIN_CODE_COMMIT`/`GEMMA4_TIERS_CHAIN_CODE_COMMIT` and the
   snapshot/revision vars from step 1 set). **Run the two chains
   sequentially, not concurrently** — both target `gpushort` and would
   otherwise compete for the same GPU allocation and double the failure
   surface for no wall-clock gain worth having. Each chain has exactly one
   writer for its output dir
   (`results/gemma4/gemma4_12b/beans_zero_greedy` and
   `results/gemma4/gemma4_12b/tiers_1_3_greedy` respectively).
