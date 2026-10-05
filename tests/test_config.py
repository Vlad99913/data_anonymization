from typing import get_args

from anonymization.anonymizer import OPERATORS, OperatorName
from anonymization.config import MODELS, ModelName


def test_model_names_match_models():
    assert set(get_args(ModelName)) == set(MODELS)


def test_operator_names_match_operators():
    assert set(get_args(OperatorName)) == set(OPERATORS)
