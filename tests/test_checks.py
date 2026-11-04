"""Answer checks: numbers, citations and database tags. No GPU, no database."""
from quillmark_bot.checks import (case_only_numbers, invalid_citations, normalize_database_tags, numbers_in,
                                  unverified_numbers)

HANDBOOK = "### [POL-ACC-02] Overdraft fee of $29 per item over $50. Refund by 11:59 p.m. on the next business day."
DATABASE = "Counted: outcomes, where issue = Goodwill request.\n- favorable: 46 cases (35% of 132)"


def test_formats_of_the_same_figure_match():
    assert numbers_in("$1,500.00 and 35% and 2.40") == {"1500", "35", "2.4"}


def test_ids_and_dates_inside_ids_are_not_figures():
    assert numbers_in("[POL-DSP-04] [INC-2026-03] [QM-C-000123]") == set()


def test_backed_figures_pass():
    answer = "The fee is $29 for overdrafts over $50 [POL-ACC-02]. 46 of 132 requests (35%) were approved."
    assert unverified_numbers(answer, [HANDBOOK, DATABASE]) == []


def test_invented_figure_is_flagged():
    answer = "According to our records, 180 cases involved workplace calls; 46 were approved."
    assert unverified_numbers(answer, [HANDBOOK, DATABASE]) == ["180"]


def test_figures_in_the_question_count_as_given():
    assert unverified_numbers("Your $286.09 charge will be reversed.", [HANDBOOK, "I was charged $286.09 twice"]) == []


def test_a_restated_date_is_not_flagged():
    answer = "This covers cases opened between January 1, 2025, and December 31, 2025."
    assert unverified_numbers(answer, ["opened 2025-01-01 to 2025-12-31"]) == []


def test_another_customers_amount_is_flagged():
    case = "Customer: my debit card purchase of $446.52 was charged twice."
    answer = "Your duplicate charge of $446.52 will be reversed, with a $25 credit."
    assert unverified_numbers(answer, [HANDBOOK, "$25 goodwill credit", case]) == []  # found somewhere...
    assert case_only_numbers(answer, [HANDBOOK, "$25 goodwill credit"], [case]) == ["$446.52"]  # ...but only there


def test_citations_must_have_been_provided():
    assert invalid_citations(["POL-ACC-02", "QM-C-000999"], {"POL-ACC-02", "QM-C-000001"}) == ["QM-C-000999"]


def test_invented_database_tags_become_one_tag():
    text = "35% [database_results], 37 cases [DATABASE-FRAUD-2025], 57 [Database results] and [case records]."
    assert normalize_database_tags(text) == "35% [case records], 37 cases [case records], 57 [case records] and [case records]."


def test_policy_citations_are_left_alone():
    assert normalize_database_tags("see [POL-DSP-04]") == "see [POL-DSP-04]"
