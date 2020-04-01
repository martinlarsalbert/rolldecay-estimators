import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
#from rolldecayestimators.simulation import simulate
from rolldecayestimators.direct_estimator import DirectEstimator
from rolldecayestimators.tests.test_estimator import check, simulator
import matplotlib.pyplot as plt

d = 0.076
T0 = 20
omega0 = 2 * np.pi / T0
zeta = 0.044

@pytest.fixture
def df_roll_decay():

    phi0 = np.deg2rad(2)
    phi1d0 = 0
    t = np.arange(0, 120, 0.01)
    estimator = DirectEstimator()
    yield estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0, omega0=omega0, d=d, zeta=zeta)

def check(X, estimator):
    estimator.fit(X=X)
    X_pred = estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

    assert_almost_equal(X['phi'].values, X_pred['phi'].values, decimal=3)
    assert estimator.score(X) > 0.999
    assert_almost_equal(estimator.parameters['zeta'], zeta, decimal=5)
    assert_almost_equal(estimator.parameters['d'], d, decimal=4)
    assert_almost_equal(estimator.parameters['omega0'], omega0, decimal=5)

def test_fit_simualtion_derivation(df_roll_decay):

    direct_estimator = DirectEstimator(fit_method='derivation', omega_regression=False)
    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator)


def test_fit_simualtion_derivation_omega(df_roll_decay):

    direct_estimator = DirectEstimator(fit_method='derivation', omega_regression=True)

    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator)

def test_fit_simualtion_integration(df_roll_decay):

    direct_estimator = DirectEstimator(fit_method='integration', omega_regression=False)

    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator)

def test_fit_simualtion_integration_omega(df_roll_decay):

    direct_estimator = DirectEstimator(fit_method='integration', omega_regression=True)

    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator)


