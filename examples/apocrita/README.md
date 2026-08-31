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
