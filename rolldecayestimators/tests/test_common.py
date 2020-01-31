import pytest

from sklearn.utils.estimator_checks import check_estimator

from rolldecayestimators import TemplateEstimator
from rolldecayestimators import RollDecayCutTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, RollDecayCutTransformer]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
