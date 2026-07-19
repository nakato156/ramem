from ramem.evaluation.prepare import _normalized_hash


def test_normalized_hash_ignores_case_and_whitespace() -> None:
    assert _normalized_hash("  Lima\nES Perú ") == _normalized_hash("lima es perú")
