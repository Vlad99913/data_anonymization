"""Command line interface to detect and anonymize sensitive entities in text files."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from .anonymizer import OPERATORS, Anonymizer, OperatorName, to_entities
from .config import MODELS, ModelName


def _existing_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    if path.is_file() and path.suffix != ".txt":
        raise argparse.ArgumentTypeError(f"expected a .txt file, got: {path}")
    return path


def _iter_files(path: Path) -> Iterator[Path]:
    yield from sorted(path.glob("*.txt")) if path.is_dir() else [path]


def _anonymize_file(
    anonymizer: Anonymizer,
    src: Path,
    operator: OperatorName
) -> tuple[list[dict], str]:
    """
    Anonymize a file line by line and return detected entities for each line.

    Args:
        anonymizer (Anonymizer): The anonymizer object.
        src (Path): The path to the source file.
        operator (OperatorName): The operator to use for anonymization.

    Returns:
        tuple[list[dict], str]: A tuple containing a list of entity blocks and the anonymized lines.
        The entity blocks are dictionaries with the following keys:
        - "line": The line number.
        - "entities": A list of dictionaries with the following keys:
            - "label": The label of the entity.
            - "start": The start index of the entity.
            - "end": The end index of the entity.
            - "text": The text of the entity.
            - "score": The score of the entity.
            - "recognizer": The recognizer that detected the entity.
    """
    blocks = []
    anonymized_lines = ""
    with src.open(encoding="utf-8") as src_file:
        for line_no, raw in enumerate(src_file, start=1):
            line = raw.rstrip("\n")
            results = anonymizer.detect(line) if line.strip() else []
            anonymized_line = anonymizer.anonymize(line, results, operator) + raw[len(line):]
            anonymized_lines += anonymized_line
            blocks.append({
                "line": line_no,
                "entities": [e.model_dump() for e in to_entities(results, line)],
            })
    return blocks, anonymized_lines


def main() -> None:
    """
    Run the CLI to detect and anonymize sensitive entities in text file or directory of text files.
    """
    parser = argparse.ArgumentParser(
        description="Detect and anonymize sensitive entities in text files."
    )
    parser.add_argument(
        "inputs",
        type=_existing_path,
        help="Text file or directory containing .txt files",
    )
    parser.add_argument("--model", default="spacy-lg", choices=sorted(MODELS))
    parser.add_argument("--operator", default="replace", choices=sorted(OPERATORS))
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--no-rules", action="store_true", help="Use the NER model only")
    args = parser.parse_args()
    # Both values are restricted by choices= above.
    model = cast(ModelName, args.model)
    operator = cast(OperatorName, args.operator)

    anonymizer = Anonymizer(model, use_rules=not args.no_rules)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in _iter_files(args.inputs):
        blocks, anonymized_lines = _anonymize_file(anonymizer, path, operator)
        (args.out / f"{path.stem}.entities.json").write_text(
            json.dumps(blocks, indent=2), encoding="utf-8"
        )
        (args.out / f"{path.stem}.anonymized.txt").write_text(
            anonymized_lines, encoding="utf-8"
        )
        n_entities = sum(len(block["entities"]) for block in blocks)
        print(f"{path} -> {n_entities} entities on {len(blocks)} lines")


if __name__ == "__main__":
    main()
