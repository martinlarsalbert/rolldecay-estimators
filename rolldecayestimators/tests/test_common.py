import pytest

from sklearn.utils.estimator_checks import check_estimator

from rolldecayestimators import DirectEstimator, CutTransformer


@pytest.mark.parametrize(
    "Estimator", [DirectEstimator, CutTransformer]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
