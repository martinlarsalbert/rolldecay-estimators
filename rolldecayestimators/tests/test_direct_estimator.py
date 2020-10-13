import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
#from rolldecayestimators.simulation import simulate
from rolldecayestimators.direct_estimator import DirectEstimator
import matplotlib.pyplot as plt

@pytest.fixture
def d():
    yield 0.076

@pytest.fixture
def omega0():
    T0 = 20.0
    yield 2.0*np.pi/T0

@pytest.fixture
def zeta():
    yield 0.044

@pytest.fixture
def df_roll_decay(omega0, d, zeta):

    phi0 = np.deg2rad(2)
    phi1d0 = 0
    t = np.arange(0, 120, 0.01)
    estimator = DirectEstimator.load(omega0=omega0, d=d, zeta=zeta)

    yield estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0)

@pytest.fixture
def df_roll_decay_negative(omega0, d, zeta):

    phi0 = -np.deg2rad(2)
    phi1d0 = 0
    t = np.arange(0, 120, 0.01)
    estimator = DirectEstimator.load(omega0=omega0, d=d, zeta=zeta)
    yield estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0)

def check(X, estimator, omega0, d, zeta, decimal=4):
    estimator.fit(X=X)
    assert estimator.result['success']
    X_pred = estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

    assert_almost_equal(X['phi'].values, X_pred['phi'].values, decimal=decimal)
    assert estimator.score(X) > 0.999
    assert_almost_equal(estimator.parameters['omega0'], omega0, decimal=decimal)
    assert_almost_equal(estimator.parameters['zeta'], zeta, decimal=decimal)
    assert_almost_equal(estimator.parameters['d'], d, decimal=decimal)


def test_fit_simualtion_derivation(df_roll_decay, omega0, d, zeta):

    direct_estimator = DirectEstimator(fit_method='derivation', omega_regression=False)
    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, omega0=omega0, d=d, zeta=zeta, decimal=2)  # Bad accuracy! 


def test_fit_simualtion_derivation_omega(df_roll_decay,omega0, d, zeta):

    direct_estimator = DirectEstimator(fit_method='derivation', omega_regression=True)

    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, omega0=omega0, d=d, zeta=zeta, decimal=2)  # Bad accuracy!

def test_fit_simualtion_integration(df_roll_decay,omega0, d, zeta):

    direct_estimator = DirectEstimator(fit_method='integration', omega_regression=False)

    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, omega0=omega0, d=d, zeta=zeta)

def test_fit_simualtion_integration_omega(df_roll_decay,omega0, d, zeta):

    direct_estimator = DirectEstimator(fit_method='integration', omega_regression=True)

    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, omega0=omega0, d=d, zeta=zeta)

def test_fit_simualtion_integration_omega_negative(df_roll_decay_negative,omega0, d, zeta):

    direct_estimator = DirectEstimator(fit_method='integration', omega_regression=True)

    X = df_roll_decay_negative
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, omega0=omega0, d=d, zeta=zeta)


