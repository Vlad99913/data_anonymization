import argparse
import re

import pytest
from presidio_analyzer import RecognizerResult

from anonymization.cli import _anonymize_file, _existing_path, _iter_files


class FakeAnonymizer:
    """Tags every "John" as PERSON without loading an NER model."""

    def __init__(self):
        self.detect_calls = []

    def detect(self, text):
        self.detect_calls.append(text)
        return [
            RecognizerResult(
                "PERSON", m.start(), m.end(), 0.9, recognition_metadata={"recognizer_name": "Fake"}
            )
            for m in re.finditer("John", text)
        ]

    def anonymize(self, text, results, operator):
        for r in sorted(results, key=lambda r: r.start, reverse=True):
            text = text[:r.start] + f"<{r.entity_type}>" + text[r.end:]
        return text


def test_existing_path_rejects_missing(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(argparse.ArgumentTypeError, match="path does not exist"):
        _existing_path(str(missing))


def test_existing_path_rejects_non_txt_file(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("x")
    with pytest.raises(argparse.ArgumentTypeError, match="expected a .txt file"):
        _existing_path(str(path))


def test_iter_files_single_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("x")
    assert list(_iter_files(path)) == [path]


def test_iter_files_directory_yields_sorted_txt_files(tmp_path):
    for name in ["b.txt", "a.txt", "c.csv"]:
        (tmp_path / name).write_text("x")
    assert list(_iter_files(tmp_path)) == [tmp_path / "a.txt", tmp_path / "b.txt"]


def test_anonymize_file_line_by_line(tmp_path):
    src = tmp_path / "input.txt"
    src.write_text("Hi John\n\n   \nJohn met John", encoding="utf-8")
    anonymizer = FakeAnonymizer()

    blocks, anonymized = _anonymize_file(anonymizer, src, "replace")

    assert anonymized == "Hi <PERSON>\n\n   \n<PERSON> met <PERSON>"
    assert anonymizer.detect_calls == ["Hi John", "John met John"]
    assert [b["line"] for b in blocks] == [1, 2, 3, 4]
    assert blocks[1]["entities"] == blocks[2]["entities"] == []
    assert blocks[0]["entities"] == [
        {
            "label": "PERSON",
            "text": "John",
            "start": 3,
            "end": 7,
            "score": 0.9,
            "recognizer": "Fake",
        },
    ]
    assert [(e["start"], e["end"]) for e in blocks[3]["entities"]] == [(0, 4), (9, 13)]
