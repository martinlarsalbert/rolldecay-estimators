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

    estimator.parameters = {
         'omega0':omega0,
         'zeta': zeta,
    }

    return estimator.simulate(t=t, phi0 = phi0, phi1d0 = phi1d0,)

def simulator_negative(estimator):
    phi0 = -np.deg2rad(2)
    phi1d0 = 0
    t = np.arange(0, 120, 0.01)

    estimator.parameters = {
        'omega0': omega0,
        'zeta': zeta,
    }

    return estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0)

def simulator_sparse(estimator):
    phi0 = np.deg2rad(2)
    phi1d0 = 0
    t = np.arange(0, 120, 10)

    estimator.parameters = {
        'omega0': omega0,
        'zeta': zeta,
    }

    df = estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0)
    return df

def simulator_noise(estimator):
    phi0 = np.deg2rad(2)
    phi1d0 = 0
    t = np.arange(0, 120, 0.1)
    estimator.parameters = {
        'omega0': omega0,
        'zeta': zeta,
    }

    df = estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0)
    np.random.seed(0)
    std = np.deg2rad(0.1)
    noise = np.random.normal(loc=0.0, scale=std, size=len(t))
    df['phi']+=noise

    return df

def check(X, estimator):
    estimator.fit(X=X)
    X_pred = estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

    assert_almost_equal(X['phi'].values, X_pred['phi'].values, decimal=3)
    assert estimator.score(X) > 0.999
    assert_almost_equal(estimator.parameters['zeta'], zeta, decimal=4)
    assert_almost_equal(estimator.parameters['omega0'], omega0, decimal=4)

def test_roll_decay_derivation_omega():

    estimator = RollDecay(fit_method='derivation', omega_regression=True)
    X = simulator(estimator=estimator)
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)

    check(X=X, estimator=estimator)

def test_roll_decay_integration_omega():

    estimator = RollDecay(fit_method='integration', omega_regression=True)
    X = simulator(estimator=estimator)
    check(X=X, estimator=estimator)

def test_roll_decay_integration_omega_negative():

    estimator = RollDecay(fit_method='integration', omega_regression=True)
    X = simulator_negative(estimator=estimator)
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

def test_roll_decay_integration_sparse():

    estimator = RollDecay(fit_method='integration', omega_regression=True)
    X = simulator_sparse(estimator=estimator)

    estimator.fit(X=X)
    X_pred = estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

def test_roll_decay_integration_noise():

    estimator = RollDecay(fit_method='integration', omega_regression=True)
    X = simulator_noise(estimator=estimator)

    estimator.fit(X=X)
    X_pred = estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

def test_load():

    estimator = RollDecay(fit_method='integration', omega_regression=False)
    X = simulator(estimator=estimator)

    data = {
        'omega0' : omega0,
        'zeta' : zeta,
    }

    estimator2 = RollDecay.load(data=data, X=X)
    X_pred = estimator2.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='loaded prediction')
    plt.show()

def test_load_score():

    estimator = RollDecay(fit_method='integration', omega_regression=False)
    X = simulator(estimator=estimator)

    data = {
        'omega0' : omega0,
        'zeta' : zeta,
    }

    estimator2 = RollDecay.load(data=data, X=X)
    assert estimator2.score(X=X) > 0.9999999999999


