# DATASHEET: ROOTS Training Dataset

## MOTIVATION

**For what purpose was the dataset created?**  
ROOTS was created to provide large-scale, structured, and diverse supervision for training bioacoustic audio-language models across a broad taxonomy of tasks aligned with real-world ethology workflows. Existing bioacoustic datasets primarily support narrow supervised tasks such as species classification and lack coverage of key capabilities such as acoustic measurement, temporal reasoning, relational inference, and in-context learning.

ROOTS addresses this gap by transforming heterogeneous bioacoustic archives and metadata into a unified audio-language training dataset. It supports multiple task families spanning acoustic perception, semantic recognition, structural and temporal reasoning, and multi-audio/in-context learning. The dataset is designed to align directly with the BEANS-Next benchmark, enabling systematic study of how different forms of supervision contribute to specific model capabilities.

**Who created this dataset and on behalf of which entity?**  
ROOTS was created by the authors of the associated paper as part of an academic research effort in bioacoustics and audio-language modeling. It is intended for use by the machine learning, computational bioacoustics, and ethology research communities.

**What support was needed to make this dataset?**  
The dataset required:
- Access to large-scale bioacoustic archives and datasets  
- Computational resources for metadata processing, audio preprocessing, and synthetic data generation  
- Language model inference for metadata-grounded text synthesis  
- Engineering effort to design and implement data pipelines for templating, synthesis, filtering, and task construction  

**Any other comments?**  
ROOTS is designed as a *training dataset*, complementary to BEANS-Next as an evaluation benchmark. A central design principle is alignment between training supervision and evaluation tasks.

---

## COMPOSITION

**What do the instances that comprise the dataset represent?**  
Each instance represents an audio-language training example. Instances may include:
- One or more audio recordings (single-audio or multi-audio)
- Structured metadata (e.g., species, taxonomy, location, call type)
- Synthetic or templated natural language prompts
- Corresponding target outputs (labels, descriptions, answers, or structured predictions)

**How many instances are there in total?**  
ROOTS is a large-scale dataset composed of:
- Millions of templated supervision examples derived from metadata  
- Large-scale synthetic examples generated through augmentation pipelines  
- Multi-audio and in-context learning examples constructed combinatorially  

Exact counts depend on the final release configuration and filtering thresholds, but the dataset is substantially larger than prior bioacoustic training datasets in both size and task diversity.

**Does the dataset contain all possible instances or is it a sample?**  
ROOTS is a curated and transformed subset of available bioacoustic data. It is not exhaustive and is biased toward:
- Datasets with accessible metadata  
- Species and regions with stronger representation in archives  
- Tasks that can be derived from available annotations or synthesized reliably  

**What data does each instance consist of?**  
Each instance consists of:
- Raw audio (or reference to audio)
- Task-specific prompt (instruction)
- Target output (label, description, numeric value, or structured answer)

Depending on the task, instances may also include:
- Multiple audio clips (for relational or in-context tasks)
- Temporal annotations (timestamps, segments)
- Synthetic labels (for detection, counting, or diarization)

**Is there a label or target associated with each instance?**  
Yes. Targets vary by task type:
- **Acoustic perception:** frequency, duration, acoustic descriptors  
- **Semantic recognition:** species, taxonomy, call type, life stage  
- **Structural/temporal:** call counts, ordering, timing, speaker counts  
- **In-context learning:** query-conditioned classification or matching  

**Is any information missing from individual instances?**  
Yes. Source datasets often contain incomplete metadata (e.g., missing call types or timestamps). Synthetic pipelines are used to fill some gaps, but coverage remains uneven across taxa and tasks.

**Are relationships between individual instances made explicit?**  
Yes, particularly in:
- Multi-audio tasks (query vs. candidates)
- Temporal tasks (event sequences within recordings)
- Relational tasks (comparisons, counting, diarization)

**Are there recommended data splits?**  
Yes. ROOTS includes training splits aligned with evaluation in BEANS-Next. Validation splits may be constructed to reflect:
- Cross-species generalization  
- Cross-dataset generalization  
- Task-level generalization  

**Are there any errors, sources of noise, or redundancies in the dataset?**  
Yes. Sources of noise include:
- Weak labels in source datasets  
- Metadata inconsistencies across datasets  
- Errors in automatic extraction pipelines  
- Imperfect synthetic generation  
- Duplicate or highly similar recordings  

Filtering and scoring pipelines reduce but do not eliminate these issues.

**Is the dataset self-contained, or does it rely on external resources?**  
ROOTS relies on external bioacoustic datasets and archives. The training dataset includes:
- Processed metadata and prompts  
- References to audio files or derived audio  

Access to raw audio may depend on source dataset licenses.

**Does the dataset contain confidential data?**  
No.

**Does the dataset contain potentially offensive content?**  
No. The dataset consists of animal vocalizations and environmental audio.

**Does the dataset relate to people?**  
No. It focuses on animal vocalizations.

**Does the dataset identify subpopulations?**  
Not for humans. For animals, it may include:
- Species  
- Taxonomic groups  
- Life stages  
- Behavioral categories  

**Is it possible to identify individuals?**  
Only in limited cases where datasets include individual animal identity labels.

**Does the dataset contain sensitive data?**  
No human-sensitive data. However, ecological sensitivity may arise from:
- Rare or endangered species  
- Precise geographic metadata  

Users should handle such data responsibly.

**Any other comments?**  
ROOTS prioritizes task diversity and supervision richness over strict dataset uniformity.

---

## COLLECTION

**How was the data acquired?**  
Data was obtained from:
- Public bioacoustic archives  
- Existing benchmark datasets  
- Curated ecological datasets  

Additional data was generated via:
- Metadata templating pipelines  
- Metadata-grounded text synthesis  
- Synthetic audio generation pipelines  

**Over what timeframe was the data collected?**  
Underlying recordings span multiple years or decades depending on source datasets. ROOTS itself was constructed during the development of the paper.

**What mechanisms were used to collect the data?**  
Original recordings were collected using:
- Field microphones  
- Autonomous recording units  
- Scientific recording campaigns  

ROOTS construction used:
- Data processing pipelines  
- Audio augmentation systems  
- Language models for text synthesis  

**What was the resource cost of collecting the data?**  
Costs include:
- Moderate compute for preprocessing and synthesis  
- Significant compute for synthetic audio generation  
- Language model inference costs for text generation  

These are substantially lower than training large foundation models but non-trivial at scale.

**If the dataset is a sample, what was the sampling strategy?**  
Sampling is:
- Task-driven (aligned with taxonomy)  
- Metadata-driven (based on available annotations)  
- Augmentation-driven (for synthetic data)  

Not random or statistically representative.

**Who was involved in the data collection process?**  
- Original dataset creators (field researchers, annotators)  
- ROOTS authors (data processing and synthesis)  

**Were ethical review processes conducted?**  
No new human-subject data was collected. Ethical considerations relate primarily to ecological impact and dataset reuse.

**Does the dataset relate to people?**  
No.

**Any other comments?**  
ROOTS inherits biases and limitations from its source datasets.

---

## PREPROCESSING / CLEANING / LABELING

**Was preprocessing/cleaning/labeling done?**  
Yes. Major steps include:
- Mapping heterogeneous metadata into a unified schema  
- Generating templated prompt-response pairs  
- Synthesizing natural language descriptions and QA pairs  
- Extracting audio features and attributes  
- Generating synthetic audio with strong labels  
- Filtering examples using consistency and quality checks  

**Was the raw data saved?**  
Raw audio remains in source datasets. ROOTS includes references and derived data.

**Is preprocessing software available?**  
Yes. Pipelines for templating, synthesis, filtering, and synthetic audio generation are intended to be released.

**Any other comments?**  
A key innovation is converting weakly labeled archives into strong supervision through structured pipelines.

---

## USES

**Has the dataset been used for any tasks already?**  
Yes. ROOTS is used to train audio-language models evaluated on BEANS-Next and other benchmarks.

**Is there a repository linking to uses?**  
A public repository is expected to accompany the release.

**What other tasks could the dataset be used for?**  
- Audio-language model training  
- Bioacoustic representation learning  
- Zero-shot and few-shot learning  
- Acoustic reasoning tasks  
- Cross-species generalization  

**Are there risks impacting future use?**  
Yes:
- Dataset bias across species and regions  
- Synthetic data artifacts  
- Label noise from weak supervision  

Mitigation includes careful evaluation and domain-specific validation.

**Are there tasks for which the dataset should not be used?**  
- High-stakes ecological decision-making without expert validation  
- Human-related inference tasks  
- Surveillance or misuse of ecological data  

**Any other comments?**  
ROOTS is designed for general-purpose training, not as a definitive ecological dataset.

---

## DISTRIBUTION

**Will the dataset be distributed externally?**  
Yes.

**How will it be distributed?**  
Likely via:
- Public repository  
- Dataset hosting platforms  

Includes:
- Metadata  
- Prompts and labels  
- Code  
- Audio references  

**When will it be distributed?**  
At or after publication.

**Will it have a license?**  
Yes. Likely a research-friendly license. Underlying datasets retain their own licenses.

**Are there third-party restrictions?**  
Yes. Source dataset licenses apply.

**Any export restrictions?**  
None known.

**Any other comments?**  
Users must comply with all upstream dataset licenses.

---

## MAINTENANCE

**Who maintains the dataset?**  
The dataset authors.

**How can they be contacted?**  
Via project repository or publication contact information.

**Is there an erratum?**  
Will be maintained via repository issues.

**Will the dataset be updated?**  
Yes. Possible updates include:
- New datasets  
- Improved synthesis pipelines  
- Bug fixes  

**Will older versions be maintained?**  
Yes, via versioning for reproducibility.

**Can others contribute?**  
Yes. Contributions may be accepted via repository workflows, subject to validation.

**Any other comments?**  
Long-term success depends on community adoption and extension.