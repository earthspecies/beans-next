# Scoring policy

The evaluator preserves raw predictions. Answer parsing does not use the reference value to select an interpretation.

## Numerical answers

Report MAE in the target units, together with the accepted-response count and total example count.
MAE includes only responses with an interpretable numerical estimate. Accepted answers are not necessarily correct answers.

The parser accepts explicit numbers, spelled-out counts, labelled measurements, and complete ranges.
It converts explicit Hz/kHz units and uses the midpoint of a range.
It accepts decimal commas with one or two fractional digits before a physical unit and preserves thousands separators.
An explicit answer can precede an explanation or numbered list.
Formula constants, list numbers, label digits, incomplete ranges, and one-sided bounds do not count as measurements.

## Multiple choice

Use the options from each example. The parser accepts letters, exact option text, unambiguous species aliases, and explicit answer statements.
It rejects conflicting selections and letter/name contradictions.
For rescoring, retain the exact questions and option order used during inference.
Sample IDs alone do not establish that two dataset revisions have the same options.

## Multi-label tasks

Fixed-vocabulary call type uses macro-F1 across its five labels.
Each label is matched separately: `alarm call` does not imply `call`.
Unrecognized list items receive no credit and make exact-set accuracy fail.

Tier 4 detection computes F1 for each reference label, then averages across labels.
`None` denotes an empty set. It is not a positive class.
Complete-set accuracy is a separate diagnostic: `A, C` equals `C, A`, but differs from `A`.

## Summaries and captions

Summary and captioning use corpus CIDEr over the complete cleaned response and reference.
Score the full task together so that IDF uses the full reference corpus.
Preserve literal `None` references. Result tables display CIDEr multiplied by 100.
CIDEr measures textual agreement; it does not independently verify counts, frequency ranges, or temporal order.

Species F1 and structured-field scores are supplementary diagnostics.
Species frequency-range evaluation averages band IoU over reference species, with zero for omitted species.

## Saved results

`processed_predictions.jsonl` stores the question and target alongside each response.
Keep this file with `predictions.jsonl` for offline rescoring.
Use a separate output directory for rescored results.
The scorer version identifies the scoring policy in cache entries and summaries.
Inspect raw and parsed answers alongside coverage before comparing scores.
