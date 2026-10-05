"""Config for the anonymization pipeline.

- LABELS: List of entities to anonymize.
- NER_MODEL_CONFIGURATION: Configuration for the NER model.
- GLINER_ENTITY_MAPPING: Mapping of GLiNER entities to Presidio entities.
- ModelName: Names of the available models.
- MODELS: Dictionary of models to use for anonymization.
"""

from dataclasses import dataclass
from typing import Literal

# Labels for the NER model.
LABELS = [
    "PERSON", "ORG", "JOB", "EMAIL_ADDRESS", "LOCATION", "AMOUNT",
    "DATE_TIME", "UNIVERSITY", "PHONE_NUMBER", "URL", "IBAN", "SSN",
]

# Map model labels to labels used for this project.
NER_MODEL_CONFIGURATION = {
    "model_to_presidio_entity_mapping": {
        "PERSON": "PERSON", "PER": "PERSON",
        "ORG": "ORG",
        "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION",
        "DATE": "DATE_TIME", "TIME": "DATE_TIME",
        "MONEY": "AMOUNT",
    },
    "labels_to_ignore": [
        "O", "MISC", "NORP", "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT",
        "EVENT", "LANGUAGE", "LAW", "PRODUCT", "WORK_OF_ART",
    ],
    "low_score_entity_names": [],
    "aggregation_strategy": "simple",
    "alignment_mode": "expand",
}

# GLiNER is zero-shot: keys are the natural-language prompts sent to the model.
GLINER_ENTITY_MAPPING = {
    "person": "PERSON",
    "organization": "ORG",
    "profession": "JOB",
    "university": "UNIVERSITY",
    "location": "LOCATION",
    "street address": "LOCATION",
    "date": "DATE_TIME",
    "time": "DATE_TIME",
    "monetary amount": "AMOUNT",
}


@dataclass(frozen=True)
class ModelSpec:
    nlp_configuration: dict
    gliner_model: str | None = None  # when set, GLiNER replaces the NLP engine's NER recognizer


# Keep in sync with the keys of MODELS (checked by tests/test_config.py).
ModelName = Literal["spacy-lg", "spacy-trf", "gliner-pii"]

MODELS: dict[ModelName, ModelSpec] = {
    "spacy-lg": ModelSpec(
        {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
        }
    ),
    "spacy-trf": ModelSpec(
        {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_trf"}]
        }
    ),
    "gliner-pii": ModelSpec(
        {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        },
        gliner_model="urchade/gliner_multi_pii-v1"
    ),
}
