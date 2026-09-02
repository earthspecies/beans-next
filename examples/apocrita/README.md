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
