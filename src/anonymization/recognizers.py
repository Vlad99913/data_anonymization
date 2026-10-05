"""Build rules-based pattern recognizers for Presidio.

Some entity types are not supported by default for certain NER models.
Regex pattern recognizers are used to handle these cases.
These detected entities are:
- AMOUNT (monetary amount)
- DATE_TIME (date and time)
- UNIVERSITY (university name)
- JOB (job title)
- PHONE_NUMBER (phone number)
- URL
- IBAN (international bank account number)
- SSN (social security number)
"""

import re

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    IbanRecognizer,
    PhoneRecognizer,
    UrlRecognizer,
)

REGEX_FLAGS_CASE_SENSITIVE = re.DOTALL | re.MULTILINE

PHONE_REGIONS = ("BE", "NL", "FR", "DE", "GB", "US")

JOB_TITLES = sorted(
    {
        "software engineer", "data scientist", "data engineer", "data analyst",
        "machine learning engineer", "project manager", "product manager",
        "engineer", "developer", "analyst", "consultant", "accountant",
        "lawyer", "attorney", "doctor", "nurse", "teacher", "professor",
        "architect", "designer", "manager", "CEO", "CTO", "CFO", "COO",
    },
    key=len,
    reverse=True,  # longest first so "software engineer" wins over "engineer"
)

UNIVERSITY_REGEX = (
    r"\b(?:[A-Z][\w&.'-]*\s+){0,4}"
    r"(?:University|College|Law School|Business School|Institute of Technology|"
    r"Universiteit|Hogeschool)\b"
    r"(?:\s+of(?:\s+[A-Z][\w.-]*){1,4})?"
)

AMOUNT_REGEX = (
    r"(?:[$€£¥]|\b(?:USD|EUR|GBP)\s?)\d+(?:[.,\u00a0 ]\d{3})*(?:[.,]\d+)?"
    r"(?:\s?(?:thousand|million|billion|trillion|bn|k|m)\b)?"
    r"|\b\d+(?:[.,]\d{3})*(?:[.,]\d+)?\s?(?:€|EUR|USD|euros?|dollars?)(?!\w)"
)

DATETIME_REGEX = (
    r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?\b"
    r"|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"
)

YEAR_REGEX = r"\b(?:19|20)\d{2}\b"


class SsnRecognizer(PatternRecognizer):
    """Belgian national register number (checksum-validated) and US SSN."""

    PATTERNS = [
        Pattern("be_national_number", r"\b\d{2}\.?\d{2}\.?\d{2}[-.\s]?\d{3}\.?\d{2}\b", 0.5),
        Pattern("us_ssn", r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b", 0.6),
    ]
    CONTEXT = [
        "ssn",
        "social security",
        "national number",
        "national register",
        "rijksregisternummer",
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="SSN",
            name="SsnRecognizer",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        """
        Validate the result of a SSN pattern recognizer.

        Args:
            pattern_text (str): The text to validate.

        Returns:
            bool | None: True if the result is valid, False if the result is invalid,
            or None if the result is not a SSN.
        """
        digits = re.sub(r"\D", "", pattern_text)
        if len(digits) != 11:
            return None
        base, check = int(digits[:9]), int(digits[9:])
        # People born from 2000 onwards get a "2" prepended before the modulo.
        return check in (97 - base % 97, 97 - (2_000_000_000 + base) % 97)


def build_pattern_recognizers() -> list[EntityRecognizer]:
    """Build a list of pattern recognizers for Presidio.

    Returns:
        list[EntityRecognizer]: A list of pattern recognizers.
    """
    return [
        EmailRecognizer(),
        UrlRecognizer(),
        IbanRecognizer(supported_entity="IBAN"),
        PhoneRecognizer(supported_regions=PHONE_REGIONS),
        SsnRecognizer(),
        PatternRecognizer(
            supported_entity="AMOUNT",
            name="AmountRecognizer",
            patterns=[Pattern("amount", AMOUNT_REGEX, 0.9)],
        ),
        PatternRecognizer(
            supported_entity="DATE_TIME",
            name="DateTimeRecognizer",
            patterns=[Pattern("datetime", DATETIME_REGEX, 0.9), Pattern("year", YEAR_REGEX, 0.35)],
        ),
        PatternRecognizer(
            supported_entity="UNIVERSITY",
            name="UniversityRecognizer",
            patterns=[Pattern("university", UNIVERSITY_REGEX, 1.0)],
            global_regex_flags=REGEX_FLAGS_CASE_SENSITIVE,
        ),
        PatternRecognizer(
            supported_entity="JOB",
            name="JobTitleRecognizer",
            deny_list=JOB_TITLES,
            deny_list_score=0.8,
            context=["works as", "job", "position", "role", "title"],
        ),
    ]
