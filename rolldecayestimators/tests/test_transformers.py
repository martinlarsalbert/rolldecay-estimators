import pytest
import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from rolldecayestimators import DirectEstimator, CutTransformer
from rolldecayestimators.transformers import OffsetTransformer
from rolldecayestimators.tests.test_cubic import simulate
import matplotlib.pyplot as plt

@pytest.fixture
def data():
    parameters = {
        'B_1A': 0.3,
        'B_2A': 0.0,
        'B_3A': 0.0,
        'C_1A': 10.0,
        'C_3A': 0.0,
        'C_5A': 0.0,
    }

    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 15, 0.1)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    return X

def test_offset_transformer(data):
    X = data

    X_offset = X.copy()
    phi_offset = np.deg2rad(1.0)
    t = X_offset.index
    delta_t = t[-1] - t[0]
    d_phi = phi_offset/delta_t
    X_offset['phi']=X_offset['phi'] + d_phi*t

    trans = OffsetTransformer()

    trans.fit(X=X)

    X_trans = trans.transform(X=X)

    fig,ax=plt.subplots()
    X.plot(y='phi', ax=ax, label='true')
    X_offset.plot(y='phi', ax=ax, label='model test')
    X_trans.plot(y='phi', ax=ax, style='--', label='offset removed')
    ax.legend()
    plt.show()

    assert_allclose(X_trans['phi'], X['phi'], atol=0.01)