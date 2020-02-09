import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from rolldecayestimators import DirectEstimator, CutTransformer


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

@pytest.mark.skip('Write a real test later...')
def test_direct_estimator(data):
    est = DirectEstimator()
    assert est.demo_param == 'demo_param'

    est.fit(*data)
    assert hasattr(est, 'is_fitted_')

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))

@pytest.mark.skip('Write a real test later...')
def test_cut_transformer_error(data):
    X, y = data
    trans = CutTransformer()
    trans.fit(X,y)
    with pytest.raises(ValueError, match="Shape of input is different"):
        X_diff_size = np.ones((10, X.shape[1] + 1))
        trans.transform(X_diff_size)


@pytest.mark.skip('Write a real test later...')
def test_cut_transformer(data):
    X, y = data
    trans = CutTransformer()
    assert trans.start == 0
    assert trans.stop == 1

    trans.fit(X,y)
    assert trans.n_features_ == X.shape[1]

    X_trans = trans.transform(X)
    assert_allclose(X_trans, X)

    X_trans = trans.fit_transform(X,y)
    assert_allclose(X_trans, X)
