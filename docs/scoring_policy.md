# Deterministic scoring policy (2026-09-25-v4)

Predictions must be interpreted without using the correct answer to choose a meaning. Preserve raw predictions and rescore into a new directory. Never score historical letter answers against a newer dataset's shuffled options.

- Numeric tasks: preserve the raw value. Convert explicit Hz/kHz units, use a range's midpoint, and reject multiple distinct measurements. Do not snap to target values or infer a scale from the reference value. Report numeric parse coverage beside conditional MAE.
- Summary (supplementary species diagnostic): score the set of named species independently of counts and frequency formatting. Use the bundled common/scientific-name aliases. Invalid output receives zero F1 on valid targets. An empty set requires explicit absence. Unknown-only targets are excluded from named-species F1 and reported through target coverage. This metric does not assess counts, frequencies, or order.
- MCQs: use each question's own options. Accept letters, exact option text, unambiguous species aliases, and explicit answer statements. Reject conflicting selections and letter/name contradictions. If original options are unavailable, only explicit letters can be recovered; a missing mapping is an evidence gap, not proof that a name answer is wrong.
- Multi-audio detection: use `multilabel_detection`. Compute F1 separately for each reference label, then average. Infer the label vocabulary from targets, as in BEANS-Zero; `None` is an all-zero row, not a separate positive class. Complete-set accuracy remains a diagnostic. `A, C` equals `C, A`, but does not equal `A`.
- Main Summary metric: use the captioning task type and corpus CIDEr over the complete raw/cleaned response and reference. Include all 1,000 saved targets, preserving literal None and unknown annotations. This measures text agreement, not numerical correctness. Keep species F1 and structured-field metrics as separate diagnostics.
- Fixed-vocabulary call type: use `multilabel_classification` and macro-F1 across the five advertised labels. Match each label separately, so `alarm call` does not imply `call`. Unknown list items mark imperfect parsing without erasing the recognized items. Unknown items receive no credit; exact-set accuracy fails. The main macro-F1 concerns the five advertised labels only.

`processed_predictions.jsonl` now retains `question` for future offline MCQ rescoring. Old artifacts may not contain it. Supplement them only with the exact original questions, never a different release with matching sample IDs. The scorer version separates corrected scored-cache entries from old scores; it does not retroactively update completed run files.

The focused tests include hand-calculated expectations, target independence, option permutation, scientific/common aliases, contradictory answers, complete multi-label sets, and online/offline answer preservation. Deterministic parsing remains conservative on unrestricted prose; inspect raw-to-parsed examples and coverage before publishing a replacement table.


## Numerical answer review (2026-09-25, scoring v4)

Numerical MAE is conditional on an accepted numerical response; always report the accepted count and full evaluated denominator. Parse success is not answer correctness. Accept explicit count clauses (including spelled-out counts and zero), labelled measurements, spelled-out physical units, and decimal commas with one or two fractional digits before a physical unit. Preserve thousands separators. Extract an explicit answer independently of targets, including when followed by list numbering or separately labelled formants. Reject formula constants, list numbers, label digits such as F1, incomplete ranges, and one-sided numerical bounds. Complete ranges retain the documented midpoint convention.

Remaining rejected responses include both model non-answers and unresolved/ambiguous formats; coverage must not be described as a perfect assessment of all human-readable answers. The older text-only SNR runs have 2,011 examples, whereas the supplied new audio predictions have 397, with no shared sample IDs. Rescoring does not resolve that dataset mismatch.
