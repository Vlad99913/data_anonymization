import pytest

from anonymization.evaluation import MODES, Counts, Evaluator, Span

# One document exercising every matching case.
GOLD = [[
    Span(0, 4, "PERSON"),
    Span(10, 15, "ORG"),
    Span(20, 25, "LOCATION"),
    Span(30, 35, "DATE_TIME"),
]]
PRED = [[
    Span(0, 4, "PERSON"),      # identical
    Span(10, 15, "LOCATION"),  # same interval, wrong label
    Span(21, 27, "LOCATION"),  # overlap only
    Span(40, 45, "URL"),       # extra prediction; DATE_TIME is missed
]]


def _totals(counts: Counts) -> tuple[int, int, int]:
    return counts.tp, counts.fp, counts.fn


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (Span(0, 5, "A"), Span(3, 8, "B"), True),
        (Span(3, 8, "B"), Span(0, 5, "A"), True),
        (Span(0, 5, "A"), Span(5, 8, "B"), False),
        (Span(5, 8, "B"), Span(0, 5, "A"), False),
        (Span(0, 10, "A"), Span(2, 4, "B"), True),
    ],
)
def test_span_overlaps(a, b, expected):
    assert a.overlaps(b) is expected


def test_counts_metrics():
    counts = Counts(tp=3, fp=1, fn=2)
    assert counts.precision == pytest.approx(0.75)
    assert counts.recall == pytest.approx(0.6)
    assert counts.f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_counts_metrics_without_data():
    counts = Counts()
    assert (counts.precision, counts.recall, counts.f1) == (0.0, 0.0, 0.0)


def test_evaluator_totals_per_mode():
    evaluator = Evaluator(GOLD)
    counts = evaluator.evaluate_model("toy", PRED)

    totals = {mode: _totals(counts[mode]["ALL"]) for mode in MODES}
    assert totals == {"strict": (1, 3, 3), "exact": (2, 2, 2), "partial": (3, 1, 1)}


def test_evaluator_per_label_counts():
    strict = Evaluator(GOLD).evaluate_model("toy", PRED)["strict"]
    assert _totals(strict["PERSON"]) == (1, 0, 0)
    assert _totals(strict["ORG"]) == (0, 0, 1)
    assert _totals(strict["LOCATION"]) == (0, 2, 1)
    assert _totals(strict["DATE_TIME"]) == (0, 0, 1)
    assert _totals(strict["URL"]) == (0, 1, 0)


def test_evaluator_rejects_invalid_mode():
    with pytest.raises(ValueError):
        Evaluator(GOLD, evaluation_modes=("strickt",))


def test_evaluator_rejects_document_count_mismatch():
    with pytest.raises(ValueError):
        Evaluator(GOLD).evaluate_model("toy", [])


def test_counts_accessors():
    evaluator = Evaluator(GOLD)
    counts = evaluator.evaluate_model("toy", PRED)
    assert evaluator.counts("toy") is counts
    assert evaluator.counts() == {"toy": counts}


def test_summary_df():
    evaluator = Evaluator(GOLD)
    evaluator.evaluate_model("a", PRED, ms_per_doc=1.0)
    evaluator.evaluate_model("b", GOLD, ms_per_doc=2.0)
    summary = evaluator.summary_df()

    assert list(summary.columns) == [
        "model_name", "evaluation_mode", "precision", "recall", "f1", "ms_per_doc",
    ]
    assert len(summary) == 2 * len(MODES)
    perfect = summary[summary.model_name == "b"]
    assert (perfect.f1 == 1.0).all()


def test_per_label_dfs_fill_missing_labels():
    evaluator = Evaluator(GOLD)
    evaluator.evaluate_model("a", PRED)
    evaluator.evaluate_model("b", GOLD)
    tables = evaluator.per_label_dfs()

    assert list(tables) == list(MODES)
    strict = tables["strict"].set_index("model_name")
    assert strict.loc["a", "URL"] == 0.0
    assert strict.loc["b", "URL"] == "-"
    assert strict.loc["b", "PERSON"] == 1.0


def test_to_markdown_sections():
    evaluator = Evaluator(GOLD)
    evaluator.evaluate_model("toy", PRED)
    report = evaluator.to_markdown("Toy benchmark")

    assert report.startswith("# Toy benchmark (1 docs, 4 entities)")
    assert "## Overall" in report
    for mode in MODES:
        assert f"## {mode.capitalize()}-match F1 per label" in report
