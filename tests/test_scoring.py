"""The evaluation's own answer checks. No GPU, no database."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from scoring import answer_problems  # noqa: E402


def test_a_figure_must_match_whole():
    item = {"must": ["$20"]}
    assert answer_problems(item, "The late fee is $20.50.", []) == ["missing '$20'"]
    assert answer_problems(item, "The late fee is $20.", []) == []
    assert answer_problems(item, "It costs $20, the lesser of $35 or 5%.", []) == []


def test_thousands_and_percentages():
    assert answer_problems({"must": ["35%"]}, "35% of 132 were approved", []) == []
    assert answer_problems({"must": ["35%"]}, "135% growth", []) == ["missing '35%'"]
    assert answer_problems({"must": ["1,500"]}, "a balance of $1,500 or more", []) == []


def test_phrases_still_match_as_text():
    assert answer_problems({"must": [["fix-it", "next business day"]]}, "Under the Fix-It Window...", []) == []
