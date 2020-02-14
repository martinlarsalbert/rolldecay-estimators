import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal

from rolldecayestimators.simulation_linear import simulate
from rolldecayestimators.direct_linear_estimator import DirectLinearEstimator
import matplotlib.pyplot as plt

@pytest.fixture
def df_roll_decay():

    phi0 = np.deg2rad(2)
    phi1d0 = 0
    T0 = 20
    omega0 = 2 * np.pi / T0
    zeta = 0.044
    N = 1000
    t = np.linspace(0, 120, N)
    yield simulate(t=t, phi0=phi0, phi1d0=phi1d0, omega0=omega0, zeta=zeta)

def test_fit_simualtion(df_roll_decay):

    direct_estimator = DirectLinearEstimator()

    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)

    direct_estimator.fit(X=X)
    X_pred = direct_estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

    assert_almost_equal(X['phi'].values, X_pred['phi'].values, decimal=4)
    assert direct_estimator.score(X) > 0.99999
