"""Evaluate anonymization performance using span-level metrics."""

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from .anonymizer import Anonymizer
from .config import MODELS, ModelName
from .evaluation import Evaluator, Span

TAG_RE = re.compile(r"<([A-Z_]+)>")


@dataclass
class Sample:
    text: str
    template: str
    spans: list[Span]


def _parse_gold(text: str, template: str) -> list[Span]:
    """Recover gold spans by aligning an anonymized template with the original text.

    Args:
        text (str): The original text.
        template (str): The anonymized template.

    Returns:
        list[Span]: A list of spans in the original text.
    """
    parts = TAG_RE.split(template)
    literals, tags = parts[0::2], parts[1::2]
    pattern = "".join(
        re.escape(literal) + ("(.+?)" if i < len(tags) else "")
        for i, literal in enumerate(literals)
    )
    match = re.fullmatch(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Template does not align with text: {template!r}")
    return [Span(match.start(i + 1), match.end(i + 1), tag) for i, tag in enumerate(tags)]


def _load_dataset(path: Path) -> list[Sample]:
    """Load the benchmark dataset from a CSV file.

    Args:
        path (Path): The path to the CSV file.

    Returns:
        list[Sample]: A list of samples from the dataset.
    """
    df = pd.read_csv(path, sep=";", dtype=str)
    samples = []
    for row in df.itertuples():
        text, template = str(row.text), str(row.label)
        samples.append(Sample(text, template, _parse_gold(text, template)))
    return samples


def _run_benchmark(dataset: Path, model_names: list[ModelName], out_dir: Path) -> None:
    """Run the benchmark for a given dataset and model names.

    Args:
        dataset (Path): The path to the dataset.
        model_names (list[ModelName]): The names of the models to benchmark.
        out_dir (Path): The path to the output directory.
    """
    references = _load_dataset(dataset)
    evaluator = Evaluator([ref.spans for ref in references])
    for model_name in model_names:
        for use_rules in (False, True):
            anonymizer = Anonymizer(model_name, use_rules=use_rules)
            full_model_name = f"{model_name} + rules" if use_rules else model_name
            anonymizer.detect(references[0].text)  # warm-up, excluded from timing

            start = time.perf_counter()
            results = [anonymizer.detect(ref.text) for ref in references]
            ms_per_doc = (time.perf_counter() - start) * 1000 / len(references)

            preds = [[Span(r.start, r.end, r.entity_type) for r in doc] for doc in results]
            same_pred_and_gold_num = sum(
                anonymizer.anonymize(ref.text, r) == ref.template
                for ref, r in zip(references, results)
            )

            evaluator.evaluate_model(
                full_model_name,
                preds,
                same_docs=f"{same_pred_and_gold_num}/{len(references)}",
                ms_per_doc=round(ms_per_doc, 1),
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator.summary_df().to_csv(out_dir / f"{dataset.stem}_summary.csv", index=False)
    report = evaluator.to_markdown(f"Benchmark on `{dataset.name}`")
    (out_dir / f"{dataset.stem}.md").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Presidio NER setups for anonymization.")
    parser.add_argument("--data", type=Path, default=Path("data/benchmark.csv"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["spacy-lg", "spacy-trf", "gliner-pii"], choices=sorted(MODELS)
    )
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    model_names = cast(list[ModelName], args.models)  # restricted by choices=
    _run_benchmark(args.data, model_names, args.out)


if __name__ == "__main__":
    main()
