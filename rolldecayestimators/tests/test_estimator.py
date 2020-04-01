import pytest
import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt

from rolldecayestimators.estimator import RollDecay

T0 = 20
omega0 = 2 * np.pi / T0
zeta = 0.044

def simulator(estimator):
    phi0 = np.deg2rad(2)
    phi1d0 = 0
    t = np.arange(0, 120, 0.01)
    return estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0, omega0=omega0, zeta=zeta)

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
    assert_almost_equal(estimator.parameters['omega0'], omega0, decimal=5)

def test_roll_decay_derivation_omega():

    estimator = RollDecay(fit_method='derivation', omega_regression=True)
    X = simulator(estimator=estimator)
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)

    check(X=X, estimator=estimator)

def test_roll_decay_integration_omega():

    estimator = RollDecay(fit_method='integration', omega_regression=True)
    X = simulator(estimator=estimator)
    check(X=X, estimator=estimator)

def test_roll_decay_derivation_no_omega():

    estimator = RollDecay(fit_method='derivation', omega_regression=False)
    X = simulator(estimator=estimator)
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)

    check(X=X, estimator=estimator)

def test_roll_decay_integration_no_omega():

    estimator = RollDecay(fit_method='integration', omega_regression=False)
    X = simulator(estimator=estimator)
    check(X=X, estimator=estimator)