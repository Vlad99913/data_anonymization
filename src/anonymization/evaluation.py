"""Span-level evaluation of predicted entities against gold annotations."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, NamedTuple

import pandas as pd

MODES = ("strict", "exact", "partial")


class Span(NamedTuple):
    start: int
    end: int
    label: str

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


class Evaluator:
    """Score predicted spans of several models against the same gold spans.

    Counts of true positives, false positives and false negatives are kept
    per model, evaluation mode and label.
    Counts can be returned as raw `Counts`, as summary / per-label DataFrames,
    or as a markdown report.
    """

    def __init__(
        self,
        gold_docs: list[list[Span]],
        evaluation_modes: tuple[str, ...] = MODES,
    ) -> None:
        """
        Args:
            gold_docs (list[list[Span]]): Gold spans for each document.
            evaluation_modes (tuple[str, ...]):
            Evaluation modes to compute. Defaults to all `MODES`.

        Raises:
            ValueError: If any of the modes is invalid.
        """
        invalid = [mode for mode in evaluation_modes if mode not in MODES]
        if invalid:
            raise ValueError(f"Invalid modes {invalid}; choose from {MODES}")
        self._gold_docs = gold_docs
        self._evaluation_modes = evaluation_modes

        # dictionary structure: model_name -> evaluation_mode -> label -> Counts
        self._counts: dict[str, dict[str, dict[str, Counts]]] = {}
        self._extra: dict[str, dict] = {}  # model_name -> extra summary columns

    def evaluate_model(
        self,
        model_name: str,
        pred_docs: list[list[Span]],
        **extra: Any,
    ) -> dict[str, dict[str, Counts]]:
        """Evaluate the predictions of one model in every evaluation mode and store the counts.

        Args:
            model_name (str): Name of the model, used as row label in the tables.
            pred_docs (list[list[Span]]): Predicted spans for each document.
            **extra: Additional columns for the summary table (e.g. timing).

        Raises:
            ValueError: If the number of predicted documents differs from the gold documents.

        Returns:
            dict[str, dict[str, Counts]]: Counts per evaluation mode and label for this model.
        """
        if len(pred_docs) != len(self._gold_docs):
            raise ValueError(
                f"Got predictions for {len(pred_docs)} documents, expected {len(self._gold_docs)}"
            )
        self._counts[model_name] = {
            mode: self._evaluate(pred_docs, mode) for mode in self._evaluation_modes
        }
        self._extra[model_name] = extra
        return self._counts[model_name]

    def counts(self, model_name: str | None = None) -> dict:
        """Return the stored counts, for all models or a single one.

        Args:
            model_name (str | None): Model name to return counts for. If None, return all models.

        Returns:
            dict:
            `model_name -> evaluation_mode -> label -> Counts`, or
            `evaluation_mode -> label -> Counts` for one model.
        """
        return self._counts if model_name is None else self._counts[model_name]

    def summary_df(self) -> pd.DataFrame:
        """Overall precision, recall and F1
        per model and evaluation mode, plus any extra columns."""
        rows = []
        for model_name, by_mode in self._counts.items():
            for evaluation_mode, counts in by_mode.items():
                total = counts["ALL"]
                rows.append({
                    "model_name": model_name, "evaluation_mode": evaluation_mode,
                    "precision": round(total.precision, 3),
                    "recall": round(total.recall, 3),
                    "f1": round(total.f1, 3),
                    **self._extra[model_name],
                })
        return pd.DataFrame(rows)

    def per_label_dfs(self) -> dict[str, pd.DataFrame]:
        """Pandas DataFrames of F1 per label for every model, one per evaluation mode."""
        return {
            evaluation_mode: pd.DataFrame([
                {
                    "model_name": model_name,
                    **{
                        k: round(v.f1, 2) for k, v in by_mode[evaluation_mode].items() if k != "ALL"
                    },
                }
                for model_name, by_mode in self._counts.items()
            ]).fillna("-")
            for evaluation_mode in self._evaluation_modes
        }

    def to_markdown(self, title: str) -> str:
        """Build a markdown report with the summary table
        and sections per evaluation mode showing F1 for each label.

        Args:
            title (str): Report title; document and entity counts are appended.

        Returns:
            str: The markdown report.
        """
        n_entities = sum(map(len, self._gold_docs))
        per_label_sections = "".join(
            f"## {mode.capitalize()}-match F1 per label\n\n{df.to_markdown(index=False)}\n\n"
            for mode, df in self.per_label_dfs().items()
        )
        return (
            f"# {title} ({len(self._gold_docs)} docs, {n_entities} entities)\n\n"
            f"## Overall\n\n{self.summary_df().to_markdown(index=False)}\n\n"
            f"{per_label_sections}"
        )

    @staticmethod
    def _is_match(gold: Span, pred: Span, mode: str) -> bool:
        """Check if a gold span matches a predicted span.

        Args:
            gold (Span): The gold span.
            pred (Span): The predicted span.
            mode (str): The mode of evaluation. Available modes are:

                - "strict": Exact interval and label match.
                - "exact": Exact interval match.
                - "partial": Interval overlap.

        Raises:
            ValueError: If the mode is invalid.

        Returns:
            bool: True if the gold span matches the predicted span, False otherwise.
        """
        if mode == "strict":  # exact interval AND label match
            return gold == pred
        elif mode == "exact":  # exact interval match
            return gold.start == pred.start and gold.end == pred.end
        elif mode == "partial":  # interval overlap
            return gold.overlaps(pred)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _evaluate(self, pred_docs: list[list[Span]], mode: str) -> dict[str, Counts]:
        """Count true positives, false positives, and false negatives
        for predicted spans against the gold spans.

        Args:
            pred_docs (list[list[Span]]): A list of predicted spans for each document.
            mode (str): The mode of evaluation.

        Returns:
            dict[str, Counts]: A dictionary of counts for each label
            and a total count for all labels.
        """
        counts: dict[str, Counts] = defaultdict(Counts)
        for gold, pred in zip(self._gold_docs, pred_docs):
            unmatched = list(pred)
            for g in gold:
                hit = next((p for p in unmatched if self._is_match(g, p, mode)), None)
                if hit is None:
                    counts[g.label].fn += 1
                else:
                    unmatched.remove(hit)
                    counts[g.label].tp += 1
            for p in unmatched:
                counts[p.label].fp += 1
        total = Counts(
            sum(c.tp for c in counts.values()),
            sum(c.fp for c in counts.values()),
            sum(c.fn for c in counts.values()),
        )
        return {**counts, "ALL": total}
