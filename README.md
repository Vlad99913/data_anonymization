# Data anonymization

Detect and anonymize sensitive information in text by combining pre-trained named entity recognition (NER) models with validated, rule-based recognizers. The pipeline is built on [Microsoft Presidio](https://microsoft.github.io/presidio/) and ships with:

- a **CLI** to anonymize text files line by line,
- a **FastAPI** endpoint,
- a **Streamlit** demo app,
- a **benchmark** that compares several NER models on a labelled dataset.

## Supported entities

| Entity | Label | Detected by |
|---|---|---|
| Person names | `PERSON` | NER model |
| Organisation | `ORG` | NER model |
| Profession | `JOB` | Job-title keyword list (GLiNER also predicts it) |
| Email address | `EMAIL_ADDRESS` | Presidio email recognizer |
| Location (country, city, street) | `LOCATION` | NER model |
| Amount | `AMOUNT` | NER model + currency regex |
| Date and time | `DATE_TIME` | NER model + date/time regex |
| University | `UNIVERSITY` | Regex (GLiNER also predicts it) |
| Phone number | `PHONE_NUMBER` | Presidio phone recognizer (BE, NL, FR, DE, UK, US) |
| Website | `URL` | Presidio URL recognizer |
| IBAN | `IBAN` | Presidio IBAN recognizer (checksum-validated) |
| Social security number | `SSN` | Belgian national register number (checksum-validated) and US SSN |

## Models

| Name | Model | Notes |
|---|---|---|
| `spacy-lg` | spaCy `en_core_web_lg` | CNN trained on OntoNotes; fast |
| `spacy-trf` | spaCy `en_core_web_trf` | RoBERTa trained on OntoNotes |
| `gliner-pii` | [`urchade/gliner_multi_pii-v1`](https://huggingface.co/urchade/gliner_multi_pii-v1) | Zero-shot model prompted with entity names |

Every model can run with or without the rule-based recognizers.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs the exact versions from `uv.lock`, including the pinned spaCy models. The GLiNER weights are downloaded from Hugging Face the first time `gliner-pii` is used.

## Usage

Run all commands from the project root.

### CLI

```bash
uv run anonymize data/test.txt --model spacy-lg --operator replace --out output
```

- The input is a single `.txt` file or a directory of `.txt` files. Files are processed line by line.
- For each input file, it writes:
  - `output/<name>.anonymized.txt`: the anonymized text, with the same lines as the input,
  - `output/<name>.entities.json`: one block per input line, listing each entity's label, start and end offsets within that line, text, score and recognizer.

| Argument | Description | Default |
|---|---|---|
| `inputs` | Path to a `.txt` file or a directory of `.txt` files to process | *(required)* |
| `--model` | NER setup: `spacy-lg`, `spacy-trf`, or `gliner-pii` | `spacy-lg` |
| `--operator` | How to transform each span: `replace`, `redact`, or `hash` | `replace` |
| `--out` | Directory for `.anonymized.txt` and `.entities.json` outputs | `output` |
| `--no-rules` | Disable rule-based recognizers; use the NER model only | off (rules enabled) |

Example output line:

```text
<PERSON> works for <ORG> as a <JOB>. His email address is <EMAIL_ADDRESS>. He recently traveled to <LOCATION> for a conference.
```

### API

```bash
uv run uvicorn anonymization.api:app --reload
```

- `GET /models` lists the available models.
- `POST /anonymize` takes `text`, `model` (default `spacy-lg`), `use_rules` (default `true`) and `operator` (default `replace`), and returns the anonymized text and the detected entities.
- Interactive docs are available at <http://localhost:8000/docs>.

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith works for Apple Inc. as a software engineer.", "model": "spacy-lg"}'
```

FastAPI was chosen because it validates requests and responses from the type hints (via Pydantic) and generates the interactive Swagger UI automatically, which doubles as a demo.

### Streamlit demo

```bash
uv run streamlit run src/anonymization/app.py
```

Choose the model, whether to use the rules, and the operator in the sidebar, type a text and click **Anonymize**. The app shows the anonymized text, a table of detected entities, and the original text with each entity highlighted.

### Benchmark

```bash
uv run anonymization-benchmark --data data/benchmark.csv --models spacy-lg spacy-trf gliner-pii --out reports
```

| Argument | Description | Default |
|---|---|---|
| `--data` | Path to the benchmark CSV (`;`-separated, with `text` and `label` columns) | `data/benchmark.csv` |
| `--models` | One or more models to evaluate (`spacy-lg`, `spacy-trf`, `gliner-pii`); each is run with and without rules | `spacy-lg` `spacy-trf` `gliner-pii` |
| `--out` | Directory for the markdown report and summary CSV | `reports` |

This writes `reports/<dataset>.md` (the full report) and `reports/<dataset>_summary.csv` (the overall table).

### Tests

```bash
uv run pytest
```

The unit tests cover the pure logic: gold-span parsing, evaluation metrics and matching modes, overlap resolution, the pattern recognizers and validators, and line-by-line file processing in the CLI. They don't load any NER model, so they run in a few seconds and work offline.

Type checking and linting are configured in `pyproject.toml`:

```bash
uv run mypy
uv run ruff check
```

## Evaluation

- **Dataset**: `data/benchmark.csv` contains 10 documents (`;`-separated). The `label` column is the expected anonymized text, for example `<PERSON> works for <ORG>...`. The gold entity spans are recovered by aligning each template with its original text.
- **Setups**: each model is evaluated with and without the rule-based recognizers.
- **Matching modes**:
  - `strict`: same interval and same label,
  - `exact`: same interval, any label,
  - `partial`: overlapping interval, any label.
- **Metrics**: precision, recall and F1 overall and per label. `same_docs` counts documents whose anonymized output matches the expected text exactly, and `ms_per_doc` is the average detection time per document.

## Results
Results were produced on:

| OS | CPU | RAM | Python | Inference |
|---|---|---|---|---|
| Ubuntu 24.04 (Linux 6.8, x86_64) |  Intel Core i9-13980HX (24 cores / 32 threads) | 32 GB | 3.12.3 (`uv` environment from `uv.lock`) | CPU only (CUDA not available) |


Results of `strict` evaluations on test dataset:

| model name         |   precision |   recall |    f1 | same pred and ref   |   ms / doc   |
|:-------------------|------------:|---------:|------:|:--------------------|-------------:|
| spacy-lg           |       0.891 |    0.804 | 0.845 | 2/10                |          3.4 |
| spacy-lg + rules   |       0.922 |    0.922 | 0.922 | 4/10                |          4.2 |
| spacy-trf          |       0.889 |    0.784 | 0.833 | 3/10                |         20.8 |
| spacy-trf + rules  |       0.92  |    0.902 | 0.911 | 5/10                |         20.6 |
| gliner-pii         |       0.75  |    0.647 | 0.695 | 1/10                |         25.3 |
| gliner-pii + rules |       0.756 |    0.667 | 0.708 | 1/10                |         10.7 |

Key findings:

- **Rules matter**: they add about 8 points of strict F1 to the spaCy models, because spaCy has no email, phone, URL, job-title or university labels.
- **`spacy-lg + rules` is the best trade-off**: the highest strict F1 at about 4 ms per document. `spacy-trf + rules` finds slightly more entities (partial F1) but is about 4 times slower.
- **GLiNER** covers all labels on its own, but loses mostly on `LOCATION` and `ORG` boundaries.
- **Some errors come from the labels themselves**: `Apple Inc` is labelled without its full stop in one row and with it in another, the area code "212" is labelled as `LOCATION`, and "Eiffel Tower" is not labelled at all.

The report also contains per-label F1 tables for each matching mode.

## Limitations

- The benchmark has only 10 documents, so scores are noisy.
- The rule-based recognizers were written while looking at the benchmark, and there is no held-out set, so the "+ rules" scores are optimistic. In particular, JOB and UNIVERSITY reach F1 1.0 largely because the patterns cover the examples in the dataset:
  - job titles come from a short fixed keyword list, which misses most real titles and flags generic words such as "manager" or "doctor" in any context;
  - the year pattern matches any 19xx or 20xx number.
- Speed (`ms_per_doc`) comes from a single pass over 10 short documents after one warm-up call, on whatever hardware ran the benchmark. Treat it as a rough relative comparison, not a measurement: for example, `gliner-pii + rules` comes out faster than `gliner-pii` alone, even though the rules only add work.
- The CLI processes text line by line, so the models lose context across line breaks.
- The benchmark dataset contains no IBAN or SSN examples.
- English only.

## Project structure

```text
.
├── data/                      # default folder for benchmark dataset and text files
├── reports/                   # default folder for generated benchmark reports
├── output/                    # default folder for CLI output
├── src/anonymization/
│   ├── config.py              # labels, NER label mappings, model setups
│   ├── recognizers.py         # rule-based Presidio pattern recognizers
│   ├── analyzer.py            # builds the Presidio AnalyzerEngine per model
│   ├── anonymizer.py          # Anonymizer: detection, overlap resolution, anonymization
│   ├── evaluation.py          # Evaluator: span matching, metrics and report
│   ├── benchmark.py           # benchmark entry point
│   ├── cli.py                 # command line interface
│   ├── api.py                 # FastAPI app
│   └── app.py                 # Streamlit demo
├── tests/                     # unit tests (pytest)
├── pyproject.toml
└── uv.lock
```
