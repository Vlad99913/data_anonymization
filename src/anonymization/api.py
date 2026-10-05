"""FastAPI implementation of the anonymization API."""

from functools import lru_cache
from importlib.metadata import version

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .anonymizer import Anonymizer, Entity, OperatorName, to_entities
from .config import MODELS, ModelName

app = FastAPI(title="Data anonymization API", version=version("anonymization"))


class AnonymizeRequest(BaseModel):
    """Request body for the anonymization API."""
    text: str = Field(
        ..., min_length=1, examples=["John Smith works for Apple Inc. as a software engineer."]
    )
    model: ModelName = "spacy-lg"
    use_rules: bool = True
    operator: OperatorName = "replace"


class AnonymizeResponse(BaseModel):
    """Response body for the anonymization API."""
    anonymized: str
    entities: list[Entity]


@lru_cache(maxsize=2 * len(MODELS))
def _get_anonymizer(model: ModelName, use_rules: bool) -> Anonymizer:
    return Anonymizer(model, use_rules=use_rules)


@app.get("/models")
def list_models() -> list[str]:
    return sorted(MODELS)


@app.post("/anonymize", response_model=AnonymizeResponse)
def anonymize(request: AnonymizeRequest) -> AnonymizeResponse:
    """
    Anonymize text using the specified model and operator.

    Args:
        request(AnonymizeRequest): The request body containing
        the text to anonymize and the model to use.

    Returns:
        AnonymizeResponse: The anonymized text and the entities detected in the text.
    """
    anonymizer = _get_anonymizer(request.model, request.use_rules)
    results = anonymizer.detect(request.text)
    return AnonymizeResponse(
        anonymized=anonymizer.anonymize(request.text, results, request.operator),
        entities=to_entities(results, request.text),
    )
