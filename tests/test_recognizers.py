import pytest
from presidio_analyzer import EntityRecognizer

from anonymization.recognizers import SsnRecognizer, build_pattern_recognizers

RECOGNIZERS: dict[str, EntityRecognizer] = {
    r.name: r for r in build_pattern_recognizers()  # type: ignore[has-type]
}


def _detect(name: str, text: str) -> list[str]:
    recognizer = RECOGNIZERS[name]
    # The base class types nlp_artifacts as required, but these recognizers don't use it.
    results = recognizer.analyze(
        text, entities=recognizer.supported_entities, nlp_artifacts=None  # type: ignore[arg-type]
    )
    return [text[r.start:r.end] for r in results]


@pytest.mark.parametrize(
    "number, expected",
    [
        ("82.05.30-025.56", True),
        ("82.05.30-025.57", False),
        ("05.05.30-025.20", True),  # born in 2000 or later
        ("123-45-6789", None),      # US SSN: no checksum to validate
    ],
)
def test_ssn_validate_result(number, expected):
    assert SsnRecognizer().validate_result(number) is expected


def test_ssn_recognizer_drops_invalid_checksum():
    assert _detect("SsnRecognizer", "My national number is 82.05.30-025.56.") == ["82.05.30-025.56"]
    assert _detect("SsnRecognizer", "My national number is 82.05.30-025.57.") == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Google paid $1.65 billion for it.", "$1.65 billion"),
        ("The fee is €50.000 per year.", "€50.000"),
        ("It costs 100 EUR in total.", "100 EUR"),
    ],
)
def test_amount_recognizer(text, expected):
    assert _detect("AmountRecognizer", text) == [expected]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Logged at 2022-12-27 08:26:49.21 by the server.", "2022-12-27 08:26:49.21"),
        ("Signed on 27/12/2022 in Brussels.", "27/12/2022"),
        ("She was born in 1994.", "1994"),
    ],
)
def test_datetime_recognizer(text, expected):
    assert _detect("DateTimeRecognizer", text) == [expected]


@pytest.mark.parametrize("name", ["Harvard University", "Harvard Law School"])
def test_university_recognizer(name):
    assert _detect("UniversityRecognizer", f"She studied at {name} for years.") == [name]


def test_university_recognizer_is_case_sensitive():
    assert _detect("UniversityRecognizer", "she went to a university nearby.") == []


def test_job_title_recognizer_prefers_longest_title():
    assert _detect("JobTitleRecognizer", "He works as a software engineer.") == [
        "software engineer"
    ]


def test_iban_recognizer_validates_checksum():
    assert _detect("IbanRecognizer", "Pay to BE68 5390 0754 7034 please.") == [
        "BE68 5390 0754 7034"
    ]
    assert _detect("IbanRecognizer", "Pay to BE68 5390 0754 7035 please.") == []


def test_phone_recognizer_belgian_number():
    assert _detect("PhoneRecognizer", "Call me at 015 29 58 58 tomorrow.") == ["015 29 58 58"]
