## BabyLM Challenge

### Participation Requirements

* Agents must use the provided pretraining corpus Strict (100M words)

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

### Scoring:

* Aggregate score: BLiMP and BLiMP-supplement (50%), (Super)GLUE (30%), MSGS (20%)
* Dynabench leaderboard for each track

### Hardware Constraints

* 1 A100 80GB GPU
* 80GB of RAM
* 500GB of Disk

### Time Constraints
* 24 Hour Time Limit

