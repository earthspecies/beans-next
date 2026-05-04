# LLM-as-Judge: Gemma 4 26B A4B (AWQ 4-bit)

beans-next supports an alternative evaluation path where a language model scores raw
model outputs directly, bypassing the label-parsing/fuzzy-matching pipeline.

Model used: [`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit)

## Model specs

| Property | Value |
|---|---|
| Base model | google/gemma-4-26B-A4B-it |
| Architecture | Mixture-of-Experts (MoE) |
| Total parameters | 25.2B |
| Active parameters | 3.8B |
| Layers | 30 |
| Experts | 8 active / 128 total + 1 shared |
| Context length | 256K tokens |
| Quantization | AWQ 4-bit |
| Supported modalities | Text (used text-only as judge) |
| License | Apache 2.0 |

## Installation

```bash
pip install -U transformers torch accelerate
# For AWQ quantization support:
pip install autoawq
```

## Serving via vLLM (recommended for predictions_v1 endpoint)

Serve the model behind the beans-next `predictions_v1` HTTP API using the
beans-next launcher. For text-only judge usage, no audio support is needed.

```bash
# Example: serve on port 8010
python -m beans_next.launcher \
  --model cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  --port 8010 \
  --dtype auto \
  --quantization awq
```

Or using vLLM directly (if your launcher wraps vLLM):

```bash
vllm serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  --dtype auto \
  --quantization awq \
  --max-model-len 8192 \
  --port 8010
```

## Inference settings (from model card)

```python
temperature = 1.0   # for open-ended tasks
top_p = 0.95
top_k = 64
# For judge YES/NO: use temperature=0.0, max_tokens=16
```

## Thinking mode

Gemma 4 supports a configurable thinking/reasoning mode:

- **Disable thinking** (recommended for judge): do not include `<|think|>` in the
  system prompt, or pass `enable_thinking=False` to `apply_chat_template`.
- **Enable thinking**: include `<|think|>` at the start of the system prompt.

For judge usage, beans-next sends a focused system prompt without `<|think|>` and
requests only 16 output tokens, which effectively suppresses extended reasoning.

## Chat template format

```python
messages = [
    {"role": "system", "content": "You are a strict evaluator..."},
    {"role": "user", "content": "Ground truth labels: cat\nModel output: \"a cat meowing\"\n\nDoes the model output correctly identify at least one of the ground truth labels? Reply YES or NO only."},
]
```

**Important for multi-turn**: do NOT include thinking content from previous turns
before the next user turn.

## Using as judge in beans-next

### During a benchmark run

Pass `--judge-url` to `beans-next run` (judge model endpoint must be running):

```bash
beans-next run \
  --predict-url http://localhost:8000/predict \
  --judge-url http://localhost:8010/predict \
  --dataset-name esc50
```

Judge outputs are written to `judge_outputs.jsonl` in the run output directory.

### Rescoring existing predictions

Re-score a `predictions.jsonl` from a previous run, adding judge scores:

```bash
beans-next score-from-file results/my-run/predictions.jsonl \
  --judge-url http://localhost:8010/predict \
  -o results/my-run-judged/
```

This writes three judge-specific files (normal rescorer files are **not** modified):

| File | Description |
|---|---|
| `judge_scored_predictions.jsonl` | All rows with `scores = {"judge_accuracy": 0.0 or 1.0}` |
| `judge_summary.json` | `RunSummary` with `metrics.mean.judge_accuracy` |
| `judge_outputs.jsonl` | Raw judge responses (sample_id, score, rationale, error) |

### Judge scoring logic

The judge receives **raw model output** (not post-processed) alongside the ground
truth labels and replies YES (→ 1.0) or NO (→ 0.0):

```
System: You are a strict evaluator for audio classification and detection tasks.
        You will be given ground truth labels and a model prediction.
        Reply only with YES if the prediction correctly identifies at least one
        ground truth label, or NO if it does not. No explanation.

User:   Ground truth labels: cat, dog
        Model output: "I can hear a cat meowing in the background"

        Does the model output correctly identify at least one of the ground truth
        labels? Reply YES or NO only.
```

The template is registered as `bioacoustic_classification_v1` in the
`beans_next.judges` template registry.

## Architecture note

`PredictV1Judge` (`beans_next.judges.predict_v1_judge`) uses the **same**
`predictions_v1` wire format as benchmark models. The judge endpoint is a
standard beans-next launcher serving Gemma 4 in text-only mode — no special
judge server required.

This differs from `JudgeScorer` (`beans_next.judges.scorer`), which uses the
separate `judge_scores_v1` wire format and expects the judge service to return
structured JSON scores directly.
