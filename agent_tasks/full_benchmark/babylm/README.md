## BabyLM Challenge: Detailed Rules and Guidelines

### Participation Requirements

* Participants must use the provided pretraining corpus
* Three tracks: Strict (100M words), Strict-Small (10M words), and Loose (100M words plus unlimited non-linguistic data)
* Loose track allowed additional non-linguistic data (e.g., speech audio, code, music, or visual input)
* Participants were encouraged to submit even if their work did not fit into any of the three tracks

### Datasets

* Pretraining corpus of approximately 100M words (Strict/Loose) or 10M words (Strict-Small)
* Sources:
* CHILDES (Child-directed speech)
* British National Corpus (dialogue portion)
* Children's Book Test
* Children's Stories Text Corpus
* Standardized Project Gutenberg Corpus
* OpenSubtitles
* QCRI Educational Domain Corpus (QED)
* Wikipedia and Simple Wikipedia
* Switchboard Dialog Act Corpus

### Evaluation Process

1. Evaluation tasks:

* BLiMP (zero-shot grammatical ability)
* (Super)GLUE (finetuned downstream task performance)
* MSGS (model inductive bias)
* BLiMP Supplement (dialogue and questions)
* Age-of-Acquisition prediction (optional)

1. Submission process:

* Upload model link and predictions to Dynabench
* Provide model predictions for each example of each task

1. Scoring:

* Aggregate score: BLiMP and BLiMP-supplement (50%), (Super)GLUE (30%), MSGS (20%)
* Dynabench leaderboard for each track

### Hardware Constraints

* The paper does not specify hardware constraints for training or evaluation
* Participants were free to use their available resources for training

