import pytest

from sklearn.utils.estimator_checks import check_estimator

from rolldecayestimators import TemplateEstimator
from rolldecayestimators import TemplateClassifier
from rolldecayestimators import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
