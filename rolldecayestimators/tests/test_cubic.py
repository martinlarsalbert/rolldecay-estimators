import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from rolldecayestimators.direct_estimator_cubic import DirectEstimatorCubic
import matplotlib.pyplot as plt

def simulate(t, phi0, phi1d0, A_44, B_1, B_2, B_3, C_1, C_3, C_5):

    estimator = DirectEstimatorCubic()
    return estimator.simulate(t=t,phi0=phi0, phi1d0=phi1d0,A_44=A_44,B_1=B_1,B_2=B_2,B_3=B_3,C_1=C_1,C_3=C_3,C_5=C_5)

def check(X, estimator, parameters, decimal=2):
    estimator.fit(X=X)
    assert estimator.result['success']
    X_pred = estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

    assert_almost_equal(X['phi'].values, X_pred['phi'].values, decimal=decimal)
    assert estimator.score(X) > 0.999

    for key,value in parameters.items():
        assert_almost_equal(estimator.parameters[key], parameters[key], decimal=decimal)

def test_simulation():

    parameters={
        'A_44':100.0,
        'B_1':0.3,
        'B_2':0.0,
        'B_3':0.0,
        'C_1':0.3,
        'C_3':0.0,
        'C_5':0.0,
    }

    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 10, 0.1)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)
    direct_estimator = DirectEstimatorCubic()
    direct_estimator.parameters=parameters
    direct_estimator.is_fitted_=True
    X_pred = direct_estimator.predict(X=X)
    assert_almost_equal(X['phi'].values, X_pred['phi'].values)

def test_fit_simualtion_derivation_cheat():

    parameters={
        'A_44':100.0,
        'B_1':0.3,
        'B_2':0.0,
        'B_3':0.0,
        'C_1':0.3,
        'C_3':0.0,
        'C_5':0.0,
    }

    p0 = parameters

    direct_estimator = DirectEstimatorCubic(fit_method='derivation',p0=p0)
    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 10, 0.1)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)


def test_fit_simualtion_derivation():

    parameters = {
        'A_44': 100.0,
        'B_1': 1.0,
        'B_2': 15,
        'B_3': 20,
        'C_1': 2,
        'C_3': 0.0,
        'C_5': 0.0,
    }

    p0 = {
        'A_44': 110.0,
        'B_1': 10.0,
        'B_2': 0.0,
        'B_3': 0.0,
        'C_1': 0.0,
        'C_3': 0.0,
        'C_5': 0.0,
    }

    direct_estimator = DirectEstimatorCubic(fit_method='derivation',p0=p0)
    phi0 = np.deg2rad(20)
    phi1d0 = 0
    t = np.arange(0, 520, 0.1)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)

def test_fit_simualtion_integration():
    parameters = {
        'A_44': 10.0,
        'B_1': 0.3,
        'B_2': 0.1,
        'B_3': 0.1,
        'C_1': 0.3,
        'C_3': 0.1,
        'C_5': 0.1,
    }

    p0 = {
        'A_44': 100.0,
        'B_1': 0.3,
        'B_2': 0.0,
        'B_3': 0.0,
        'C_1': 0.3,
        'C_3': 0.0,
        'C_5': 0.0,
    }

    direct_estimator = DirectEstimatorCubic(fit_method='integration', p0=p0)
    X = simulate(**parameters)
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)


