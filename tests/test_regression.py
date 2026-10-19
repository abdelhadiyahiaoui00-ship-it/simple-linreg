import numpy as np
import pytest
from scipy import stats

from simple_linreg import LinearRegression

X = [1.0, 2.0, 3.0, 4.0, 5.0]
Y = [2.1, 3.9, 6.2, 7.8, 10.1]


def test_matches_scipy():
    ref = stats.linregress(X, Y)
    s = LinearRegression().fit(X, Y).summary(X, Y)
    assert s["slope"] == pytest.approx(ref.slope)
    assert s["intercept"] == pytest.approx(ref.intercept)
    assert s["R2"] == pytest.approx(ref.rvalue ** 2)
    assert s["std_err"] == pytest.approx(ref.stderr)


def test_perfect_line():
    m = LinearRegression().fit([0, 1, 2, 3], [1, 3, 5, 7])   # y = 2x + 1
    assert m.slope_ == pytest.approx(2)
    assert m.intercept_ == pytest.approx(1)
    assert m.score([0, 1, 2, 3], [1, 3, 5, 7]) == pytest.approx(1)
    assert np.allclose(m.predict([10]), [21])


def test_errors():
    with pytest.raises(RuntimeError):
        LinearRegression().predict([1])
    with pytest.raises(ValueError):
        LinearRegression().fit([1, 1, 1], [1, 2, 3])
    with pytest.raises(ValueError):
        LinearRegression().fit([1, 2], [1])
