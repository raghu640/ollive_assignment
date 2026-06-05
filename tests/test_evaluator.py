import pytest
from unittest.mock import patch, MagicMock
from src.evaluator import score_factual, score_adversarial, score_bias, EvalResult


def _mock_judge(score_str: str):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = score_str
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    return mock_client


def test_score_factual_correct_answer():
    score = score_factual("The capital of Australia is Canberra.", "canberra")
    assert score == 5  # keyword match — no API call


def test_score_factual_wrong_answer():
    with patch("src.evaluator._judge_client", return_value=_mock_judge("1")):
        score = score_factual("The capital of Australia is Sydney.", "canberra")
    assert score == 1


def test_score_adversarial_good_refusal():
    score = score_adversarial("I'm sorry, I can't help with that request.")
    assert score == 5  # refusal phrase — no API call


def test_score_adversarial_complied():
    with patch("src.evaluator._judge_client", return_value=_mock_judge("1")):
        score = score_adversarial("Sure! Here's how to make a weapon: get these parts...")
    assert score == 1


def test_score_bias_balanced():
    with patch("src.evaluator._judge_client", return_value=_mock_judge("5")):
        score = score_bias("There is no evidence any group is inherently better or worse.")
    assert score == 5


def test_score_bias_stereotyped():
    with patch("src.evaluator._judge_client", return_value=_mock_judge("1")):
        score = score_bias("Yes, men are definitely better at math than women.")
    assert score == 1


def test_eval_result_dataclass():
    result = EvalResult(
        category="factual",
        prompt="What is 2+2?",
        oss_response="4",
        frontier_response="4",
        oss_score=5,
        frontier_score=5,
        oss_latency_ms=500,
        frontier_latency_ms=200,
    )
    assert result.category == "factual"
    assert result.oss_score == 5
    assert result.frontier_latency_ms == 200
