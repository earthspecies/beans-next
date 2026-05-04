# LLM-as-judge guide

beans-next supports three judge modes for tasks where deterministic metrics are insufficient.
Each mode is served differently and produces a different kind of output.

---

## Table of contents

- [When to use a judge](#when-to-use-a-judge)
- [Judge modes at a glance](#judge-modes-at-a-glance)
- [Mode 1 — JudgeScorer (judge\_scores\_v1)](#mode-1--judgescorerr-judge_scores_v1)
- [Mode 2 — PredictV1Judge (YES/NO binary scorer)](#mode-2--predictv1judge-yesno-binary-scorer)
- [Mode 3 — PredictV1Extractor (structured prediction generator)](#mode-3--predictv1extractor-structured-prediction-generator)
- [Running on existing predictions (score-from-file)](#running-on-existing-predictions-score-from-file)
- [Running as part of a pipeline (beans-next run)](#running-as-part-of-a-pipeline-beans-next-run)
- [Output artifacts reference](#output-artifacts-reference)
- [Serving Gemma 4 as judge](#serving-gemma-4-as-judge)
- [Template system](#template-system)
- [Python API](#python-api)
- [Practical notes](#practical-notes)

---

## When to use a judge

| Task type | Recommended approach |
|---|---|
| Classification (closed-vocabulary) | Deterministic (accuracy, F1) |
| Multi-label detection | Deterministic (average precision) |
| Captioning | Deterministic (SPIDEr = CIDEr + SPICE) |
| Classification (model outputs vary wildly) | **Extractor judge** → normal metrics |
| Open-ended description / short-answer QA | **JudgeScorer** or **YES/NO judge** |
| Counting ("how many vocalizations per species?") | **JudgeScorer** |

Two signals that a judge is needed:

1. **Format mismatch**: the model outputs a paragraph but the metric expects a single label. Fuzzy matching fails or scores badly for models that happen to be more verbose.
2. **Semantic equivalence**: "a European robin calling" and "Erithacus rubecula" are the same answer but string-match fails. A judge catches this; Levenshtein does not.

---

## Judge modes at a glance

| Mode | Class | Endpoint format | Output | Use when |
|---|---|---|---|---|
| **JudgeScorer** | `beans_next.judges.JudgeScorer` | `judge_scores_v1` (dedicated judge API) | scalar score `[0.0, 1.0]` | You run a custom judge server that returns structured scores |
| **PredictV1Judge** | `beans_next.judges.PredictV1Judge` | `predictions_v1` (standard predict API) | binary YES/NO → `judge_accuracy` metric | Judge model is served via the standard beans-next launcher |
| **PredictV1Extractor** | `beans_next.judges.PredictV1Extractor` | `predictions_v1` (standard predict API) | structured prediction text → normal metrics | Model outputs are too noisy to parse; judge restructures them |

**Key distinction**: `JudgeScorer` talks to a dedicated scoring service that returns JSON scores directly. `PredictV1Judge` and `PredictV1Extractor` both talk to a standard `predictions_v1` endpoint (the same format as benchmark models), so any model served by the beans-next launcher can act as a judge — no special judge server required.

---

## Mode 1 — JudgeScorer (`judge_scores_v1`)

The original judge mode. beans-next sends `(rubric, reference_text, candidate_text)` to a dedicated judge HTTP endpoint and receives back a scalar score in `[0.0, 1.0]` per sample.

### Architecture

```
beans-next run
    │
    ├──POST /predict──► inference launcher (predictions_v1)
    │                           │
    │                   raw predictions
    │                           │
    │                  post-processing
    │                           │
    ├──POST /judge───► judge service (judge_scores_v1)
    │                           │
    │                   {"score": 0.85, "rationale": "..."}
    │
    └── judge_outputs.jsonl
```

### Wire protocol

**Request (`POST <judge-url>`)**:

```json
{
  "schema_version": "judge_scores_v1",
  "items": [
    {
      "sample_id": "abc-001",
      "rubric": "You are a strict evaluator...\n\nGround-truth: ...\nCandidate: ...",
      "reference_text": "Turdus migratorius: 3",
      "candidate_text": "American Robin: 3"
    }
  ]
}
```

**Response**:

```json
{
  "schema_version": "judge_scores_v1",
  "items": [
    {
      "sample_id": "abc-001",
      "score": 0.85,
      "rationale": "Species matched by common name. Count correct.",
      "error": null
    }
  ]
}
```

- `score` must be in `[0.0, 1.0]`; `null` when `error` is set.
- Response items may be in any order; beans-next matches by `sample_id`.

### Built-in rubric templates

**`bioacoustic_open_qa_v1`** — general bioacoustic open-answer tasks.
- Rewards correct species, call types, behaviours, habitat, temporal patterns.
- Penalises hallucinated species and contradictions.
- Does not credit answers about speech or music unless the reference concerns them.

**`bioacoustic_counting_v1`** — vocalization counting tasks.
- Accepts any formatting: JSON, YAML, tables, bullet lists, prose.
- Accepts common names when they unambiguously identify a reference species.
- Scores refusals explicitly as 0.0.
- Awards partial credit per correct species+count pair.

### CLI usage

```bash
beans-next run \
  --predict-url http://127.0.0.1:8000/predict \
  --judge-url   http://127.0.0.1:8010/judge
```

### Implementing a judge service

Any HTTP server that accepts `judge_scores_v1` payloads and returns scores works.
Minimal FastAPI example using the Anthropic API:

```python
import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
client = anthropic.Anthropic()

@app.post("/judge")
async def judge(request: Request):
    body = await request.json()
    out_items = []
    for item in body["items"]:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": item["rubric"]}],
        )
        text = msg.content[0].text.strip()
        try:
            score = float(text.split()[0])
            score = max(0.0, min(1.0, score))
            rationale = " ".join(text.split()[1:])
            error = None
        except (ValueError, IndexError):
            score = None
            rationale = None
            error = f"Could not parse score: {text!r}"
        out_items.append({
            "sample_id": item["sample_id"],
            "score": score,
            "rationale": rationale,
            "error": error,
        })
    return JSONResponse({"schema_version": "judge_scores_v1", "items": out_items})
```

---

## Mode 2 — PredictV1Judge (YES/NO binary scorer)

`PredictV1Judge` uses the **standard `predictions_v1` predict endpoint** — the same wire format used by benchmark models — to ask the judge model a binary question: did the model prediction correctly identify at least one of the ground truth labels?

No custom judge server is needed. Serve any supported model via the beans-next launcher and point `--judge-url` at its predict endpoint.

### How it works

For each non-error sample, beans-next constructs a two-turn chat:

```
System:
  You are a strict evaluator for audio classification and detection tasks.
  You will be given ground truth labels and a model prediction.
  Reply only with YES if the prediction correctly identifies at least one
  ground truth label, or NO if it does not. No explanation.

User:
  Ground truth labels: cat, dog
  Model output: "I can hear what sounds like a cat meowing loudly."

  Does the model output correctly identify at least one of the ground truth
  labels? Reply YES or NO only.
```

The judge model replies `YES` → score 1.0, or `NO` → score 0.0. This score is stored as `judge_accuracy` in the output files.

The judge receives the **raw model output** (not post-processed), so it sees exactly what the model said — including any hedging, verbosity, or unusual phrasing that trips up fuzzy matching.

### Data flow

```
predictions.jsonl  (raw model outputs)
        │
        ▼
PredictV1Judge.score_batch()
  - POST /predict → judge model (text-only, no audio)
  - parse YES → 1.0 / NO → 0.0
        │
        ▼
judge_scored_predictions.jsonl   scores = {"judge_accuracy": 0.0 or 1.0}
judge_summary.json               metrics.mean.judge_accuracy
judge_outputs.jsonl              raw judge responses (sample_id, score, rationale, error)
```

### When to use

- Model outputs are verbose / formatted differently across model families, so fuzzy matching gives inconsistent results.
- You want a quick accuracy signal without the overhead of full structured extraction.
- You want to compare models where one is clean/structured (fuzzy matching works) and another is verbose (fuzzy matching fails).

### Template

The default user message is rendered from the `bioacoustic_classification_v1` template (registered in `beans_next.judges`). Variables available in the Jinja2 body: `reference_text`, `candidate_text`, `sample_id`, `task_id`.

---

## Mode 3 — PredictV1Extractor (structured prediction generator)

`PredictV1Extractor` sends raw model outputs to a judge model and asks it to **produce a new structured prediction** in the format the scoring pipeline expects. The extracted prediction is then scored with the normal beans-next metrics (accuracy, F1, mAP, SPIDEr).

This is an alternative to the regex/fuzzy-matching post-processing pipeline for models whose outputs do not conform to the expected label format.

### How it works

Two-turn chat, task-specific template:

**Classification**:

```
System:
  You are a structured data extraction assistant. Given a raw audio model
  response below, extract the single predicted class label. Return only the
  exact label name from the provided vocabulary list — no punctuation, no
  explanation, nothing else.

User:
  Valid labels:
  cat, dog, bird, car, rain

  Model response to structure:
  I'm pretty sure I can hear a domestic cat — specifically it sounds like
  a tabby meowing at something off-screen.

  Return the single most likely label from the list above:
```

Judge responds: `cat`

**Detection**:

```
System:
  You are a structured data extraction assistant. Given a raw audio model
  response below, extract all detected sound event labels. Return only a
  comma-separated list of label names from the provided vocabulary — no
  explanation, no extra text.

User:
  Valid labels:
  cat, dog, bird, car, rain

  Model response to structure:
  The soundscape is dominated by a dog barking intermittently, with what
  sounds like light rain in the background.

  Return all detected labels from the list above (comma-separated):
```

Judge responds: `dog, rain`

**Captioning**:

```
System:
  You are a structured data extraction assistant. Given a raw audio model
  response below, extract and clean the audio description. Return only a
  single concise sentence describing the audio — no hedging, no
  meta-commentary, no extra text.

User:
  Model response to structure:
  Let me think carefully... I believe this recording features birds near
  water? It's a bit ambiguous but possibly a forest stream environment with
  songbirds.

  Cleaned audio description:
```

Judge responds: `Songbirds calling near a forest stream.`

The extracted text is then run through minimal post-processing (whitespace normalisation + EOS stripping, plus comma-split for detection) and scored with the normal pipeline. The result: **standard metrics (accuracy, F1, mAP, SPIDEr) computed on judge-extracted predictions**.

### Data flow

```
predictions.jsonl  (raw model outputs)
        │
        ▼
PredictV1Extractor.extract_batch()
  - POST /predict → judge model (text-only, no audio)
  - judge returns structured prediction per sample
        │
        ▼
normal post-process (whitespace strip, EOS strip, comma-split for detection)
        │
        ▼
normal scoring pipeline (accuracy / F1 / mAP / SPIDEr)
        │
        ▼
judge_extracted_scored_predictions.jsonl   standard metric scores
judge_extracted_summary.json               metrics.mean.{accuracy, f1, ...}
```

### When to use

- Classification / detection model outputs are too verbose or noisy for fuzzy matching.
- You want **full comparable metrics** (not just binary judge accuracy) computed on what the model actually said.
- You want to compare the extractor-path metrics against the fuzzy-matching-path metrics to see which better reflects model capability.
- Captioning model outputs include hedging, meta-commentary, or refusals that pollute SPIDEr scores.

### Template selection

`PredictV1Extractor` selects the template automatically from `task_type`:

| `task_type` contains | Template used |
|---|---|
| `"classification"` | `extraction_classification_v1` |
| `"detection"` | `extraction_detection_v1` |
| `"caption"` | `extraction_captioning_v1` |
| anything else / `None` | `extraction_classification_v1` (default) |

---

## Running on existing predictions (`score-from-file`)

This is the most common path: you already ran inference and saved `predictions.jsonl`, and now want to apply the judge without re-running inference.

### Command

```bash
beans-next score-from-file results/my-run/predictions.jsonl \
  [--judge-url          http://localhost:8010/predict] \
  [--judge-extract-url  http://localhost:8010/predict] \
  [--task-type          classification|detection|captioning] \
  [-o results/my-run-judged/]
```

`--judge-url` and `--judge-extract-url` are independent; both may be provided in the same call. Both point at a judge model's `POST /predict` endpoint; they may or may not point at the same endpoint.

### What gets written

Normal rescorer artifacts (always written):

| File | Content |
|---|---|
| `processed_predictions.jsonl` | Post-processed predictions |
| `scored_predictions.jsonl` | Scores from fuzzy/label-matching pipeline |
| `summary.json` | Summary with standard metric means |

With `--judge-url` (YES/NO judge, Mode 2):

| File | Content |
|---|---|
| `judge_scored_predictions.jsonl` | All rows; `scores = {"judge_accuracy": 0.0 or 1.0}` |
| `judge_summary.json` | `metrics.mean.judge_accuracy` |
| `judge_outputs.jsonl` | Raw judge responses (sample_id, score, rationale, error) |

With `--judge-extract-url` (extractor, Mode 3):

| File | Content |
|---|---|
| `judge_extracted_scored_predictions.jsonl` | All rows; `processed_prediction` = judge-extracted text; `scores` = standard metrics on extracted text |
| `judge_extracted_summary.json` | `metrics.mean.{accuracy, f1, ...}` on extracted predictions |

All judge artifacts use separate filenames prefixed with `judge_` or `judge_extracted_`. Normal rescorer output is never overwritten.

### Full example: three scoring paths in one call

```bash
beans-next score-from-file results/my-run/predictions.jsonl \
  --task-type           classification \
  --judge-url           http://localhost:8010/predict \
  --judge-extract-url   http://localhost:8010/predict \
  -o results/my-run-all/
```

This writes six artifact files: the three normal files plus three judge files. You can then compare:

- `summary.json` → fuzzy-matching metrics
- `judge_summary.json` → YES/NO judge accuracy
- `judge_extracted_summary.json` → standard metrics on judge-extracted predictions

### Using the Python API directly

```python
from beans_next.runner.rescorer import rescore_predictions_file

summary = rescore_predictions_file(
    "results/my-run/predictions.jsonl",
    output_dir="results/my-run-judged/",
    task_type="classification",
    judge_url="http://localhost:8010/predict",
    judge_extract_url="http://localhost:8010/predict",
)
```

Or use the convenience wrapper for extraction-only:

```python
from beans_next.runner.rescorer import judge_extract_from_predictions_file

summary = judge_extract_from_predictions_file(
    "results/my-run/predictions.jsonl",
    "http://localhost:8010/predict",
    task_type="classification",
    output_dir="results/my-run-extracted/",
)
```

---

## Running as part of a pipeline (`beans-next run`)

The YES/NO judge (Mode 2, `--judge-url`) can also run inline during `beans-next run`, after inference and standard scoring complete for each batch. Judge inputs are accumulated and the judge is called once at the end of the run.

```bash
beans-next run \
  --predict-url http://localhost:8000/predict \
  --judge-url   http://localhost:8010/predict \
  --dataset-name esc50 \
  -o results/with-judge/
```

In pipeline mode, only `judge_outputs.jsonl` is written alongside the normal artifacts. To get `judge_scored_predictions.jsonl` and `judge_summary.json`, run `score-from-file` afterwards with `--judge-url`.

> **Note**: `--judge-extract-url` is not yet supported in pipeline mode. Use `score-from-file` for extractor-mode judge scoring.

### Architecture: pipeline mode

```
beans-next run
    │
    ├── POST /predict ──► inference launcher
    │                             │
    │                     raw predictions
    │                             │
    │                    post-process (cleaner/parser steps)
    │                             │
    │                    normal scoring (fuzzy matching)
    │                             │
    │              collect (DatasetExample, raw_text) pairs
    │
    │  (after all batches complete)
    │
    └── POST /predict ──► judge model (PredictV1Judge)
                                  │
                          YES/NO per sample
                                  │
                          judge_outputs.jsonl
```

---

## Output artifacts reference

### `judge_outputs.jsonl`

Written by `JudgeScorer` (Mode 1) and `PredictV1Judge` (Mode 2). One line per judged sample:

```json
{"error": null, "rationale": "YES", "sample_id": "esc50_0042", "score": 1.0}
```

Fields:

| Field | Type | Description |
|---|---|---|
| `sample_id` | `str` | Sample identifier |
| `score` | `float \| null` | Score in `[0.0, 1.0]`; null when `error` is set |
| `rationale` | `str \| null` | Raw judge response text (useful for debugging) |
| `error` | `str \| null` | Per-item error message; null on success |

Errored inference samples are excluded from the judge batch and will not appear here.

### `judge_scored_predictions.jsonl`

Written by `PredictV1Judge` (Mode 2) during `score-from-file`. Same schema as `scored_predictions.jsonl` but with `scores` replaced by `{"judge_accuracy": <float>}`:

```json
{
  "error": null,
  "postprocess_version": null,
  "predictions": ["I can hear a cat meowing loudly."],
  "processed_prediction": "cat",
  "sample_id": "esc50_0042",
  "scores": {"judge_accuracy": 1.0},
  "targets": "cat",
  "task_id": null
}
```

### `judge_summary.json`

Written by `PredictV1Judge` (Mode 2). Standard `RunSummary` structure with `run_id = "judge-score-from-file"` and `metrics.mean.judge_accuracy`.

### `judge_extracted_scored_predictions.jsonl`

Written by `PredictV1Extractor` (Mode 3). Same schema as `scored_predictions.jsonl` but with:

- `processed_prediction`: the judge-extracted structured text (e.g. `"cat"` or `"dog, rain"`)
- `scores`: standard metrics (accuracy, F1, mAP, SPIDEr) computed on the extracted prediction

```json
{
  "error": null,
  "postprocess_version": null,
  "predictions": ["I'm pretty sure I can hear a cat meowing at something."],
  "processed_prediction": "cat",
  "sample_id": "esc50_0042",
  "scores": {"accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0, "top1_accuracy": 1.0},
  "targets": "cat",
  "task_id": null
}
```

### `judge_extracted_summary.json`

Written by `PredictV1Extractor` (Mode 3). Standard `RunSummary` with `run_id = "judge-extract-from-file"` and standard metric means over all extracted predictions.

---

## Serving Gemma 4 as judge

`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (Gemma 4 26B A4B MoE, AWQ 4-bit) is the recommended judge model for both Mode 2 and Mode 3. It runs text-only inference on modest GPU resources (3.8B active parameters from a 25.2B MoE architecture).

See **[docs/judge_model_gemma4.md](judge_model_gemma4.md)** for full serving instructions.

Quick reference:

```bash
# Install
pip install -U transformers torch accelerate autoawq

# Serve via vLLM (wraps in beans-next predictions_v1 launcher)
python -m beans_next.launcher \
  --model cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  --port 8010 \
  --dtype auto \
  --quantization awq

# Or bare vLLM (then wrap with OpenAI-compatible proxy)
vllm serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  --dtype auto \
  --quantization awq \
  --max-model-len 8192 \
  --port 8010
```

**Thinking mode**: Gemma 4 has a configurable thinking/reasoning mode. beans-next's judge prompts intentionally do not include the `<|think|>` token, and `max_tokens=16` (Mode 2) / `max_tokens=128` (Mode 3) effectively suppress extended reasoning. For extraction tasks, use temperature=0.0 (already the default).

---

## Template system

All three judge modes use Jinja2 templates to build the content sent to the judge. Templates are registered by name in `beans_next.judges.base._TEMPLATES`.

### Built-in templates

| Template ID | Mode | Purpose |
|---|---|---|
| `bioacoustic_open_qa_v1` | JudgeScorer (Mode 1) | General bioacoustic QA rubric |
| `bioacoustic_counting_v1` | JudgeScorer (Mode 1) | Vocalization counting rubric |
| `bioacoustic_classification_v1` | PredictV1Judge (Mode 2) | YES/NO user message for classification/detection |
| `extraction_classification_v1` | PredictV1Extractor (Mode 3) | Extract single label from vocab |
| `extraction_detection_v1` | PredictV1Extractor (Mode 3) | Extract comma-separated labels from vocab |
| `extraction_captioning_v1` | PredictV1Extractor (Mode 3) | Clean/extract one-sentence description |

### Available Jinja2 variables

All templates have access to:

| Variable | Content |
|---|---|
| `reference_text` | Ground truth labels joined as a string (empty for captioning) |
| `candidate_text` | Raw model output |
| `sample_id` | Sample identifier |
| `task_id` | Task identifier |

### Listing templates

```python
from beans_next.judges import list_judge_templates

print(list_judge_templates())
# ['bioacoustic_classification_v1', 'bioacoustic_counting_v1',
#  'bioacoustic_open_qa_v1', 'extraction_captioning_v1',
#  'extraction_classification_v1', 'extraction_detection_v1']
```

### Registering a custom template

```python
from beans_next.judges import register_judge_template

MY_TEMPLATE = """Valid labels:
{{ reference_text }}

Model response:
{{ candidate_text }}

Is the model response correct? Reply YES or NO."""

register_judge_template("my_yes_no_v1", MY_TEMPLATE)
```

Then use it:

```python
from beans_next.judges import PredictV1Judge

judge = PredictV1Judge(
    "http://localhost:8010/predict",
    template_id="my_yes_no_v1",
)
```

---

## Python API

### `PredictV1Judge` — binary scorer

```python
from beans_next.judges import PredictV1Judge
from beans_next.api.types import DatasetExample

judge = PredictV1Judge(
    judge_url="http://localhost:8010/predict",
    template_id="bioacoustic_classification_v1",  # default
    system_prompt=None,      # uses built-in YES/NO prompt
    max_tokens=16,           # YES/NO only
    timeout=120.0,
    max_attempts=3,
)

examples = [
    DatasetExample(sample_id="s1", labels="cat", metadata={}),
    DatasetExample(sample_id="s2", labels=["dog", "rain"], metadata={}),
]
raw_texts = [
    "I can clearly hear a cat meowing.",
    "There's definitely a dog barking here.",
]

results = judge.score_batch(examples, raw_texts)
for r in results:
    print(r.sample_id, r.score, r.error)
# s1  1.0  None
# s2  1.0  None
```

### `PredictV1Extractor` — structured prediction generator

```python
from beans_next.judges import PredictV1Extractor
from beans_next.api.types import DatasetExample

extractor = PredictV1Extractor(
    judge_url="http://localhost:8010/predict",
    task_type="classification",   # selects extraction_classification_v1
    max_tokens=128,
    timeout=120.0,
    max_attempts=3,
)

examples = [
    DatasetExample(sample_id="s1", labels=["cat", "dog", "bird"], metadata={}),
]
raw_texts = [
    "I'm fairly confident I can detect a domestic cat in this recording."
]

extracted = extractor.extract_batch(examples, raw_texts)
print(extracted)
# ["cat"]
```

### `JudgeScorer` — `judge_scores_v1` endpoint

```python
from beans_next.judges import JudgeScorer
from beans_next.runner.runner import BenchmarkRunner

judge = JudgeScorer(
    "http://127.0.0.1:8010/judge",
    template_id="bioacoustic_counting_v1",
    timeout=120.0,
    max_attempts=3,
)

runner = BenchmarkRunner(client, renderer, config, judge=judge)
runner.run(examples)
```

### `rescore_predictions_file` — full rescoring with optional judge passes

```python
from beans_next.runner.rescorer import rescore_predictions_file

summary = rescore_predictions_file(
    "results/my-run/predictions.jsonl",
    output_dir="results/judged/",
    task_type="classification",
    judge_url="http://localhost:8010/predict",          # Mode 2, optional
    judge_extract_url="http://localhost:8010/predict",  # Mode 3, optional
)
```

### `judge_extract_from_predictions_file` — extraction-only convenience

```python
from beans_next.runner.rescorer import judge_extract_from_predictions_file

summary = judge_extract_from_predictions_file(
    "results/my-run/predictions.jsonl",
    "http://localhost:8010/predict",
    task_type="captioning",
    output_dir="results/extracted/",
)
```

---

## Practical notes

### Which judge mode to start with?

Start with `PredictV1Extractor` (`--judge-extract-url`) if your primary concern is format mismatch — the model gives correct answers but in a format fuzzy matching can't parse. The extractor produces **standard metrics** you can compare directly against other models.

Use `PredictV1Judge` (`--judge-url`) if you want a quick binary sanity check ("is this answer at all correct?") without the overhead of full metric computation on extracted predictions. `judge_accuracy` is easy to interpret.

Use `JudgeScorer` if you have a custom judge service that already produces `judge_scores_v1`-format responses, or for open-ended tasks (counting, QA) where neither YES/NO nor structured extraction captures the nuance.

### Retroactive vs. inline judging

All three modes support retroactive judging on `predictions.jsonl` via `score-from-file`. This is the preferred workflow because:

- You run inference once, saving `predictions.jsonl`.
- You try multiple judge configurations without re-running GPU inference.
- Normal scoring artifacts are never overwritten; judge artifacts get separate prefixed filenames.

Inline judging during `beans-next run` is supported for Mode 1 (`JudgeScorer`) and Mode 2 (`PredictV1Judge` via `--judge-url`). Mode 3 (`PredictV1Extractor`) must be run retroactively.

### Same endpoint for bench model and judge

You can point `--judge-url` / `--judge-extract-url` at the **same endpoint** as `--predict-url` if you want to use the benchmark model itself as its own judge. This is usually not useful for accuracy evaluation but can be helpful for debugging templates.

### Raw vs. processed predictions

`PredictV1Judge` and `PredictV1Extractor` always receive the **raw model output** (`predictions[0]`), not the post-processed text. This is intentional: the judge sees exactly what the model said, including any verbosity or formatting that tripped up fuzzy matching.

### Error handling

Per-item errors from the judge endpoint (including parse failures for Mode 2) are recorded in the output files with `error` set and `score = null`. Errored inference samples are never sent to the judge. A judge error on a sample does not count as a scoring error — the normal score is preserved, and the judge score is null.

### Parallelism

`PredictV1Extractor.extract_batch()` and `PredictV1Judge.score_batch()` send all samples in a single HTTP request to the judge model. For large datasets (>10,000 samples), consider batching manually or using `score-from-file` in chunks.

### Cost and latency

Gemma 4 26B A4B with AWQ 4-bit runs with 3.8B active parameters. At `max_tokens=16` (Mode 2), a batch of 1,000 samples typically completes in under a minute. At `max_tokens=128` (Mode 3), expect 5–10 minutes for 1,000 samples on a single A100.

### Comparing judge paths

A recommended comparison workflow:

```bash
# 1. Run inference once
beans-next run \
  --predict-url http://localhost:8000/predict \
  --dataset-name esc50 \
  -o results/esc50/

# 2. Normal rescoring (fuzzy matching)
# (artifacts already in results/esc50/ from step 1)

# 3. YES/NO judge
beans-next score-from-file results/esc50/predictions.jsonl \
  --task-type classification \
  --judge-url http://localhost:8010/predict \
  -o results/esc50-judge/

# 4. Extractor judge
beans-next score-from-file results/esc50/predictions.jsonl \
  --task-type classification \
  --judge-extract-url http://localhost:8010/predict \
  -o results/esc50-extracted/

# Compare:
# results/esc50/summary.json                    → fuzzy-matching metrics
# results/esc50-judge/judge_summary.json        → judge_accuracy
# results/esc50-extracted/judge_extracted_summary.json → standard metrics on extracted preds
```
