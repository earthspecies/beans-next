# DATASHEET: BEANS-Next Benchmark

## MOTIVATION

**For what purpose was the dataset created?**  
BEANS-Next was created to evaluate audio-language models on a broader and more realistic set of bioacoustic tasks than existing benchmarks. Prior work in bioacoustics has focused primarily on narrow recognition tasks such as species identification. However, real-world ethology workflows require a wider range of abilities, including acoustic measurement, temporal reasoning, relational analysis, call counting, comparison, and task generalization.

The dataset addresses this gap by introducing a taxonomy of bioacoustic audio-language abilities and organizing evaluation tasks into four tiers: acoustic perception, semantic recognition, structural and relational reasoning, and multi-audio/in-context learning. The goal is to provide a more comprehensive benchmark for assessing whether audio-language models can support scientific workflows in ethology, biodiversity monitoring, conservation, and animal communication research.

**Who created this dataset and on behalf of which entity?**  
The dataset was created by the authors of the BEANS-Next benchmark as part of an academic research effort in bioacoustics and audio-language modeling. It is intended for use by the broader machine learning, computational bioacoustics, and ethology communities.

**What support was needed to make this dataset?**  
The dataset builds on existing publicly available bioacoustic datasets and benchmarks, including BEANS-Zero and BirdSet, and required computational resources for data integration, metadata normalization, prompt construction, and benchmark formatting. No large-scale new annotation campaign was required for the benchmark, since labels are derived from existing ground-truth annotations.

**Any other comments?**  
A key design principle is that all labels originate from ground-truth annotations. Models are used only for metadata reformatting or templating, not for generating supervision signals.

## COMPOSITION

**What do the instances that comprise the dataset represent?**  
Instances represent bioacoustic recordings paired with structured annotations and task-specific prompts. Depending on the task, an instance may include a single audio recording, multiple audio recordings, associated biological or acoustic metadata, and a natural language instruction or query.

**How many instances are there in total?**  
The benchmark aggregates multiple underlying datasets and newly constructed evaluation tasks. The exact number of instances depends on the released configuration and task subset. Instances are organized across multiple task families spanning acoustic perception, semantic recognition, structural and relational reasoning, and multi-audio/in-context learning.

**Does the dataset contain all possible instances or is it a sample?**  
The dataset is a curated sample from the much larger universe of available bioacoustic recordings. It is not intended to be statistically representative of all animal sounds or global biodiversity. Instead, it is designed to cover a broad range of bioacoustic audio-language abilities, taxa, acoustic conditions, and task formats relevant to ethology.

**What data does each instance consist of?**  
Each instance may consist of raw audio or a reference to an audio file, ground-truth annotations, task-specific metadata, and an instruction-following input/output format. Depending on the task, the target may be a categorical label, numeric acoustic measurement, count, temporal ordering, duration estimate, comparison, or query-conditioned answer.

**Is there a label or target associated with each instance?**  
Yes. Labels depend on the task tier. Tier 1 tasks target acoustic properties such as fundamental frequency, duration, or signal characteristics. Tier 2 tasks target semantic categories such as species, call type, taxonomic group, life stage, or environmental sound class. Tier 3 tasks target structural and relational outputs such as call counts, individual counts, event timing, duration, ordering, or cross-call comparisons. Tier 4 tasks target multi-audio or in-context outputs, such as matching an audio query to candidate recordings or adapting to a task specified at inference time.

**Is any information missing from individual instances?**  
Some instances may lack certain metadata, depending on the source dataset. For example, some recordings may have species labels but no precise timestamps, call-type annotations, or acoustic measurements. Missing information reflects limitations of the original datasets rather than intentional removal.

**Are relationships between individual instances made explicit?**  
Yes, for tasks where relationships are part of the evaluation. In temporal tasks, relationships between calls within a recording may be represented through timestamps, orderings, or segment boundaries. In multi-audio and in-context tasks, relationships between query examples and candidate recordings are made explicit through task structure.

**Are there recommended data splits?**  
Yes. The benchmark provides recommended evaluation splits. Where possible, these inherit or adapt the original splits from source datasets such as BEANS-Zero and BirdSet. The splits are designed to support evaluation of generalization across tasks, taxa, recording conditions, and query formats.

**Are there any errors, sources of noise, or redundancies in the dataset?**  
Yes. Potential sources of noise include annotation errors inherited from source datasets, ambiguity in call-type or species labels, environmental background noise, overlapping vocalizations, and recording-condition variation. These sources of noise are partly intentional, since they reflect realistic bioacoustic settings.

**Is the dataset self-contained, or does it link to or otherwise rely on external resources?**  
The dataset relies on external bioacoustic datasets and benchmarks, including BEANS-Zero, BirdSet, and other source archives or datasets used to construct the evaluation tasks. Availability, licensing, and long-term stability of the underlying audio depend on the original sources. The benchmark layer consists of task definitions, prompts, metadata transformations, splits, and evaluation scripts.

**Does the dataset contain data that might be considered confidential?**  
No. The dataset is composed of animal vocalizations and environmental audio from research datasets or public archives.

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?**  
No. The audio consists of animal sounds and environmental recordings. Some recordings may include loud, harsh, or noisy sounds, but they are not expected to be offensive or threatening.

**Does the dataset relate to people?**  
No. The dataset focuses on animal vocalizations and environmental sound. It is not intended to contain human speech or human-subject data.

**Does the dataset identify any subpopulations?**  
Not applicable. The dataset does not relate to people. For animals, it may identify biological categories such as species, taxonomic group, life stage, call type, or individual identity when such labels are available.

**Is it possible to identify individuals from the dataset?**  
Not natural persons. In some tasks, individual animals may be identifiable if the source dataset includes individual identity annotations, but the dataset is not designed to identify humans.

**Does the dataset contain sensitive data?**  
No sensitive human data is included. Some ecological location metadata in source datasets may be sensitive if it concerns rare or endangered species. Users should follow source-dataset restrictions and avoid using the dataset to expose vulnerable species or habitats.

**Any other comments?**  
The dataset is designed for ecological and scientific validity rather than exhaustive taxonomic or geographic representativeness.

## COLLECTION

**How was the data associated with each instance acquired?**  
The data was acquired from existing bioacoustic datasets and benchmarks with ground-truth annotations. Audio recordings were originally collected through field recordings, autonomous recording units, curated archives, or prior benchmark datasets. BEANS-Next derives task instances from these sources by normalizing metadata and converting annotations into instruction-following evaluation examples.

**Over what timeframe was the data collected?**  
The underlying recordings span the collection periods of the source datasets. These periods may vary by archive, species, geographic region, and recording campaign. BEANS-Next itself was constructed during the development of the associated paper.

**What mechanisms or procedures were used to collect the data?**  
Original data collection mechanisms include microphones, autonomous recording units, field recording protocols, and archive submissions. Benchmark construction used software scripts for metadata parsing, label normalization, task generation, split construction, and evaluation formatting.

**What was the resource cost of collecting the data?**  
BEANS-Next primarily required computational resources for processing, formatting, and validation. The benchmark construction cost is expected to be small compared with the cost of training large audio-language models. The original source datasets may have involved substantial fieldwork, recording equipment, annotation effort, and institutional support.

**If the dataset is a sample from a larger set, what was the sampling strategy?**  
Sampling was task-driven rather than purely random. Instances were selected or transformed to cover the benchmark taxonomy and to evaluate a broad range of abilities. The design prioritizes diversity of task type and scientific relevance rather than proportional representation of taxa or geographic regions.

**Who was involved in the data collection process and how were they compensated?**  
The original data collection and annotation were performed by the creators of the source datasets, including bioacoustics researchers, field biologists, annotators, archive contributors, and dataset curators. BEANS-Next was constructed by the benchmark authors. Compensation details for source-dataset contributors depend on the original datasets and are not controlled by BEANS-Next.

**Were any ethical review processes conducted?**  
No separate human-subject ethical review was required for BEANS-Next because the benchmark uses non-human animal audio from existing datasets. Ethical review, permits, or fieldwork approvals for original recordings are inherited from the source datasets where applicable.

**Does the dataset relate to people?**  
No.

**Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources?**  
Not applicable for human subjects. The benchmark obtains animal audio and metadata from third-party source datasets and archives.

**Were the individuals in question notified about the data collection?**  
Not applicable.

**Did the individuals in question consent to the collection and use of their data?**  
Not applicable.

**If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?**  
Not applicable.

**Has an analysis of the potential impact of the dataset and its use on data subjects been conducted?**  
No human-subject impact analysis is required. Ecological impacts should nevertheless be considered, especially for endangered species or sensitive locations. Users should comply with source-dataset restrictions and avoid uses that could facilitate harm to wildlife.

**Any other comments?**  
BEANS-Next inherits both strengths and limitations from its source datasets, including taxonomic imbalance, geographic bias, and variation in annotation protocols.

## PREPROCESSING / CLEANING / LABELING

**Was any preprocessing, cleaning, or labeling done?**  
Yes. Benchmark construction involved standardizing metadata, normalizing labels where necessary, converting source annotations into task-specific targets, generating natural language prompts, and organizing examples into benchmark tiers and task families. Audio may also be segmented, resampled, or referenced according to task requirements.

**Was the raw data saved in addition to the preprocessed/cleaned/labeled data?**  
The raw audio remains available through the original source datasets or archives, subject to their access policies and licenses. BEANS-Next preserves references to source data where possible.

**Is the software used to preprocess, clean, or label the instances available?**  
The benchmark construction and evaluation code is intended to be released with the benchmark, including scripts for metadata conversion, task formatting, and evaluation.

**Any other comments?**  
All task labels are derived from ground-truth annotations in the source datasets. Models are not used to create labels, except possibly for non-label transformations such as metadata reformatting or prompt templating.

## USES

**Has the dataset been used for any tasks already?**  
Yes. BEANS-Next is used in the associated paper to evaluate existing bioacoustic models and general audio-language models across the proposed taxonomy of abilities. It is also used to evaluate the effect of training on the authors’ proposed bioacoustic audio-language dataset.

**Is there a repository that links to any or all papers or systems that use the dataset?**  
A public repository or benchmark page is intended to accompany the release. It should include dataset documentation, task definitions, evaluation scripts, and links to associated papers or systems.

**What other tasks could the dataset be used for?**  
The dataset could be used for evaluating audio-language models, bioacoustic representation learning, zero-shot and few-shot bioacoustic classification, audio retrieval, multi-audio reasoning, acoustic measurement, temporal reasoning, and task-conditioned evaluation of animal vocalization models.

**Is there anything about the composition, collection, or preprocessing that might impact future uses?**  
Yes. The dataset may contain taxonomic, geographic, and recording-condition biases inherited from source datasets. Some task families may be better represented than others. Users should avoid interpreting benchmark performance as evidence of universal competence across all taxa, environments, or ethological tasks. Users should also be careful when applying models trained or evaluated on this benchmark to rare species, endangered species, or high-stakes conservation decisions.

**Are there tasks for which the dataset should not be used?**  
The dataset should not be used for human-related inference tasks, surveillance, or applications requiring exhaustive biodiversity coverage. It should not be used as the sole basis for high-stakes ecological or conservation decisions without expert validation. It should also not be used to infer sensitive locations of vulnerable species if source metadata could expose them.

**Any other comments?**  
BEANS-Next is primarily an evaluation benchmark. It is not intended to replace expert bioacoustic analysis, ecological field validation, or taxon-specific model evaluation.

## DISTRIBUTION

**Will the dataset be distributed to third parties outside of the entity on behalf of which it was created?**  
Yes. The benchmark is intended for distribution to the research community.

**How will the dataset be distributed?**  
The benchmark will likely be distributed through a public repository and/or dataset hosting platform. The release may include metadata, task definitions, prompts, splits, evaluation scripts, and links or references to source audio where redistribution of raw audio is not permitted.

**When will the dataset be distributed?**  
The benchmark is intended to be released at or after publication of the associated paper.

**Will the dataset be distributed under a copyright or other IP license and/or terms of use?**  
The benchmark layer, including task definitions, prompts, splits, and evaluation code, is expected to be released under a permissive research license. The underlying audio remains governed by the licenses and terms of use of the original source datasets.

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**  
Yes. Some source datasets may impose licensing, citation, redistribution, or access restrictions. Users must comply with all applicable licenses and terms from the original data providers.

**Do any export controls or other regulatory restrictions apply?**  
No known export controls apply. Users should nevertheless comply with source-dataset terms and any relevant biodiversity, wildlife, or data-sharing restrictions.

**Any other comments?**  
Because the benchmark aggregates or references external datasets, users should verify licensing compatibility before redistributing audio or using the benchmark for commercial purposes.

## MAINTENANCE

**Who is supporting, hosting, and maintaining the dataset?**  
The benchmark authors are expected to support and maintain the initial release.

**How can the owner, curator, or manager be contacted?**  
Contact information should be provided through the associated paper, project website, or public repository.

**Is there an erratum?**  
An erratum or issue tracker should be maintained through the public repository.

**Will the dataset be updated?**  
Potentially. Updates may include bug fixes, corrected labels, improved metadata mappings, additional task instances, additional source datasets, or expanded evaluation tiers. Updates should be versioned and communicated through repository releases.

**If the dataset relates to people, are there applicable limits on retention?**  
Not applicable. The dataset does not relate to people.

**Will older versions continue to be supported, hosted, or maintained?**  
Where feasible, older versions should remain available through versioned releases to support reproducibility.

**If others want to extend, augment, build on, or contribute to the dataset, is there a mechanism for them to do so?**  
Yes. Contributions may be accepted through a public repository, such as pull requests or issue submissions. New tasks or datasets should be reviewed for consistency with the BEANS-Next taxonomy, label quality, licensing compatibility, and scientific relevance.

**Any other comments?**  
Long-term maintenance will depend on community adoption and continued use by the bioacoustics and audio-language modeling communities.