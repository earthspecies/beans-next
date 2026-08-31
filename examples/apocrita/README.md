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
