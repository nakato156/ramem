from ramem.evaluation.prepare import _has_blocking_overlap, _normalized_hash


def test_normalized_hash_ignores_case_and_whitespace() -> None:
    assert _normalized_hash("  Lima\nES Perú ") == _normalized_hash("lima es perú")


def test_isolated_question_match_is_not_blocking() -> None:
    assert not _has_blocking_overlap(contexts=0, pairs=0)
    assert _has_blocking_overlap(contexts=1, pairs=0)
    assert _has_blocking_overlap(contexts=0, pairs=1)
