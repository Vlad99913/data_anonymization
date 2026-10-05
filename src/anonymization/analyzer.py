"""Build an analyzer engine to perform NER for a given model."""

from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngine, NlpEngineProvider
from presidio_analyzer.predefined_recognizers import GLiNERRecognizer

from .config import GLINER_ENTITY_MAPPING, MODELS, NER_MODEL_CONFIGURATION, ModelName
from .recognizers import build_pattern_recognizers


@lru_cache
def _nlp_engine(model_name: ModelName) -> NlpEngine:
    configuration = {
        **MODELS[model_name].nlp_configuration,
        "ner_model_configuration": NER_MODEL_CONFIGURATION
    }
    return NlpEngineProvider(nlp_configuration=configuration).create_engine()


@lru_cache
def _gliner_recognizer(model_name: str) -> GLiNERRecognizer:
    return (
        GLiNERRecognizer(
            model_name=model_name,
            entity_mapping=GLINER_ENTITY_MAPPING,
            threshold=0.4
        )
    )


def build_analyzer_engine(model: ModelName, use_rules: bool = True) -> AnalyzerEngine:
    """Build an analyzer engine to perform NER for a given model.

    Args:
        model (ModelName): The name of the model to use for NER.
        use_rules (bool): Whether to use rules-based pattern recognizers.

    Returns:
        AnalyzerEngine: An analyzer engine configured to perform NER for the given model.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODELS)}")
    spec = MODELS[model]

    # Create the NLP engine for the model
    nlp_engine = _nlp_engine(model)

    # Create regex pattern recognizers if requested
    registry = RecognizerRegistry(
        recognizers=build_pattern_recognizers() if use_rules else [],
        supported_languages=["en"],
    )
    if spec.gliner_model:
        # Add GLiNER recognizer to the registry
        registry.add_recognizer(_gliner_recognizer(spec.gliner_model))
        # Remove the spaCy recognizer to avoid NER coming from spaCy
        registry.remove_recognizer("SpacyRecognizer")
    else:
        registry.add_nlp_recognizer(nlp_engine)

    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])
