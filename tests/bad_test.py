import pytest


@pytest.mark.skip(reason="Intentional failing test kept for demo; do not enable.")
def test_true_equals_false():
    assert True == False
