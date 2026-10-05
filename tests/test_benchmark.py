import pytest

from anonymization.benchmark import Sample, _load_dataset, _parse_gold
from anonymization.evaluation import Span


def _texts(text: str, spans: list[Span]) -> list[tuple[str, str]]:
    return [(text[s.start:s.end], s.label) for s in spans]


def test_parse_gold_recovers_each_tag():
    text = "Google has acquired YouTube for $1.65 billion."
    spans = _parse_gold(text, "<ORG> has acquired <ORG> for <AMOUNT>.")
    assert _texts(text, spans) == [
        ("Google", "ORG"), ("YouTube", "ORG"), ("$1.65 billion", "AMOUNT")
    ]


def test_parse_gold_leaves_punctuation_outside_tag():
    text = "John Smith works for Apple Inc. as a software engineer."
    spans = _parse_gold(text, "<PERSON> works for <ORG>. as a <JOB>.")
    assert _texts(text, spans) == [
        ("John Smith", "PERSON"), ("Apple Inc", "ORG"), ("software engineer", "JOB")
    ]


def test_parse_gold_without_tags():
    text = "Nothing sensitive here."
    assert _parse_gold(text, text) == []


def test_parse_gold_raises_on_misaligned_template():
    with pytest.raises(ValueError):
        _parse_gold("Paris is the capital of France.", "<LOCATION> is the capital of Germany.")


def test_load_dataset(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(
        "text;label\n"
        "Paris is the capital of France.;<LOCATION> is the capital of <LOCATION>.\n"
        "Nothing here.;Nothing here.\n",
        encoding="utf-8",
    )
    samples = _load_dataset(csv)

    assert len(samples) == 2
    assert all(isinstance(s, Sample) for s in samples)
    assert samples[0].template == "<LOCATION> is the capital of <LOCATION>."
    assert _texts(samples[0].text, samples[0].spans) == [
        ("Paris", "LOCATION"), ("France", "LOCATION")
    ]
    assert samples[1].spans == []
