from presidio_analyzer import RecognizerResult

from anonymization.anonymizer import Anonymizer, Entity, to_entities


def _result(label: str, start: int, end: int, score: float = 0.5) -> RecognizerResult:
    return RecognizerResult(label, start, end, score)


def _spans(results: list[RecognizerResult]) -> list[tuple[str, int, int]]:
    return [(r.entity_type, r.start, r.end) for r in results]


def test_resolve_overlaps_identical_span_keeps_higher_score():
    kept = Anonymizer._resolve_overlaps([_result("ORG", 0, 5, 0.4), _result("PERSON", 0, 5, 0.9)])
    assert _spans(kept) == [("PERSON", 0, 5)]


def test_resolve_overlaps_contained_span_loses_despite_higher_score():
    kept = Anonymizer._resolve_overlaps([_result("PERSON", 0, 10, 0.3), _result("ORG", 2, 6, 0.99)])
    assert _spans(kept) == [("PERSON", 0, 10)]


def test_resolve_overlaps_partial_overlap_keeps_longer_span():
    kept = Anonymizer._resolve_overlaps([_result("ORG", 0, 6), _result("LOCATION", 4, 14)])
    assert _spans(kept) == [("LOCATION", 4, 14)]


def test_resolve_overlaps_keeps_disjoint_spans_sorted_by_start():
    results = [_result("DATE_TIME", 20, 30), _result("PERSON", 0, 4), _result("ORG", 4, 10)]
    kept = Anonymizer._resolve_overlaps(results)
    assert _spans(kept) == [("PERSON", 0, 4), ("ORG", 4, 10), ("DATE_TIME", 20, 30)]


def test_resolve_overlaps_empty():
    assert Anonymizer._resolve_overlaps([]) == []


def test_to_entities():
    text = "John works at Acme in Paris."
    results = [
        RecognizerResult(
            "PERSON", 0, 4, 0.85432, recognition_metadata={"recognizer_name": "SpacyRecognizer"}
        ),
        RecognizerResult("ORG", 14, 18, 0.5, recognition_metadata={}),
        RecognizerResult("LOCATION", 22, 27, 0.7),
    ]
    assert to_entities(results, text) == [
        Entity(
            label="PERSON",
            text="John",
            start=0,
            end=4,
            score=0.854,
            recognizer="SpacyRecognizer",
        ),
        Entity(label="ORG", text="Acme", start=14, end=18, score=0.5, recognizer=None),
        Entity(label="LOCATION", text="Paris", start=22, end=27, score=0.7, recognizer=None),
    ]
