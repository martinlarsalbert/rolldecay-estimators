import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from rolldecayestimators.simulation import simulate
from rolldecayestimators.direct_estimator_improved import DirectEstimatorImproved
import matplotlib.pyplot as plt

@pytest.fixture
def df_roll_decay():

    phi0 = np.deg2rad(2)
    phi1d0 = 0
    d = 0.076
    T0 = 20
    omega0 = 2 * np.pi / T0
    zeta = 0.044
    N = 1000
    t = np.linspace(0, 120, N)
    yield simulate(t=t, phi0=phi0, phi1d0=phi1d0, omega0=omega0, d=d, zeta=zeta)

def test_fit_simualtion(df_roll_decay):

    direct_estimator = DirectEstimatorImproved()

    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)

    direct_estimator.fit(X=X)
    X_pred = direct_estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

    assert_almost_equal(X['phi'].values, X_pred['phi'].values, decimal=3)
    assert direct_estimator.score(X) > 0.999

def test_fit_simualtion_bounds(df_roll_decay):

    bounds = {
        'zeta':(-np.inf,0),
        'd': (-np.inf,0),
    }

    direct_estimator = DirectEstimatorImproved(bounds=bounds)

    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)

    direct_estimator.fit(X=X)
    assert direct_estimator.parameters['zeta'] <= 0
    assert direct_estimator.parameters['d'] <= 0

