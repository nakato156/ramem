from ramem.evaluation.generation import normalize_answer, score_answer, token_f1


def test_normalize_answer_removes_citation_punctuation_and_spanish_articles() -> None:
    assert normalize_answer("La capital es Lima. [D1]") == "capital es lima"


def test_score_answer_ignores_required_citation_for_answer_quality() -> None:
    metrics = score_answer("Lima [D1]", "Lima [D1]")

    assert metrics.exact_match == 1.0
    assert metrics.token_f1 == 1.0
    assert metrics.cites_d1 == 1.0
    assert metrics.valid_citations == 1.0


def test_token_f1_gives_partial_credit() -> None:
    assert token_f1("Lima, Perú", "Lima") == 2 / 3


def test_invalid_extra_citation_is_detected() -> None:
    metrics = score_answer("Lima [D1] [D2]", "Lima [D1]")

    assert metrics.cites_d1 == 1.0
    assert metrics.valid_citations == 0.0
