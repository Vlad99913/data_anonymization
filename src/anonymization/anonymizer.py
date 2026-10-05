"""Class to perform NER and anonymization on given text."""

from typing import Literal

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from pydantic import BaseModel

from .analyzer import build_analyzer_engine
from .config import LABELS, ModelName

# Keep in sync with the keys of OPERATORS (checked by tests/test_config.py).
OperatorName = Literal["replace", "redact", "hash"]

OPERATORS: dict[OperatorName, OperatorConfig] = {
    "replace": OperatorConfig("replace"),  # default new_value is "<ENTITY_TYPE>"
    "redact": OperatorConfig("redact"),
    "hash": OperatorConfig("hash", {"hash_type": "sha256"}),
}


class Entity(BaseModel):
    """Entity detected in a text, with offsets relative to that text."""
    label: str
    text: str
    start: int
    end: int
    score: float
    recognizer: str | None

    @classmethod
    def from_result(cls, result: RecognizerResult, text: str) -> "Entity":
        return cls(
            label=result.entity_type,
            text=text[result.start:result.end],
            start=result.start,
            end=result.end,
            score=round(result.score, 3),
            recognizer=(result.recognition_metadata or {}).get("recognizer_name"),
        )


def to_entities(results: list[RecognizerResult], text: str) -> list[Entity]:
    """Convert Presidio results into `Entity` objects for the given text."""
    return [Entity.from_result(r, text) for r in results]


class Anonymizer:
    """Class to perform NER and anonymization on given text.

    Combines Presidio's AnalyzerEngine and AnonymizerEngine
    to perform NER and anonymization respectively.
    """
    def __init__(self, model: ModelName = "spacy-lg", use_rules: bool = True) -> None:
        self._analyzer_engine = build_analyzer_engine(model, use_rules)
        self._anonymizer_engine = AnonymizerEngine()

    def detect(self, text: str) -> list[RecognizerResult]:
        """
        Perform NER on the given text.

        Args:
            text (str): The text to perform NER on.

        Returns:
            list[RecognizerResult]: A list of named entities detected in the text.
        """
        results = self._analyzer_engine.analyze(text=text, language="en", entities=LABELS)

        return self._resolve_overlaps(results)

    @staticmethod
    def _resolve_overlaps(ner_results: list[RecognizerResult]) -> list[RecognizerResult]:
        """
        Resolve overlaps between entities.
        Longest span wins; identical spans are decided by score.

        The merging of partially overlapping entities is normally done
        by Presidio's AnonymizerEngine, but in that process information about
        original entity spans is lost. Such information is crucial for the benchmark.

        Overlap resolution is done after entity detection but before anonymization
        to ensure that the benchmark scores exactly what gets anonymized.

        Args:
            ner_results (list[RecognizerResult]): A list of NER results.

        Returns:
            list[RecognizerResult]: A list of NER results with overlaps resolved.
        """
        kept: list[RecognizerResult] = []
        for result in sorted(ner_results, key=lambda r: (-(r.end - r.start), -r.score, r.start)):
            if all(result.end <= k.start or result.start >= k.end for k in kept):
                kept.append(result)
        return sorted(kept, key=lambda r: r.start)

    def anonymize(
        self,
        text: str,
        results: list[RecognizerResult] | None = None,
        operator: OperatorName = "replace",
    ) -> str:
        """
        Anonymize the given text.

        Args:
            text (str): The text to anonymize.
            results (list[RecognizerResult]) | None: A list of NER results.
                If None, NER is performed on the text.
            operator (OperatorName): Operation to be applied to the entities. Defaults to "replace".

        Returns:
            str: The anonymized text. The original text is returned if no anonymization is required.
        """
        results = self.detect(text) if results is None else results
        if not results:
            return text
        return self._anonymizer_engine.anonymize(
            text=text,
            # Presidio's analyzer and anonymizer each define an identical RecognizerResult.
            analyzer_results=list(results),  # type: ignore[arg-type]
            operators={"DEFAULT": OPERATORS[operator]},
            merge_entities_with_spaces=False,
        ).text
