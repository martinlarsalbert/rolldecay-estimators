import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from rolldecayestimators.direct_estimator_cubic import EstimatorCubic, EstimatorQuadraticB, EstimatorLinear
import matplotlib.pyplot as plt
from rolldecayestimators.estimator import FitError

def simulate(t, phi0, phi1d0, **kwargs):

    estimator = EstimatorCubic.load(**kwargs)

    return estimator.simulate(t=t,phi0=phi0, phi1d0=phi1d0)

def check(X, estimator, parameters, decimal=2):

    if not estimator.is_fitted_:
        estimator.fit(X=X)

    assert estimator.result['success']
    X_pred = estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

    assert_almost_equal(X['phi'].values, X_pred['phi'].values, decimal=decimal)
    assert estimator.score(X) > 0.999


    true_parameters = pd.Series(parameters)
    predicted = pd.Series(estimator.parameters, index=true_parameters.index)
    predicted.fillna(0, inplace=True)

    # Normalize with A_44:
    #true_parameters/=true_parameters['A_44']
    #predicted/=predicted['A_44']

    pd.testing.assert_series_equal(predicted.round(decimals=2), true_parameters)

    #for key,value in parameters.items():
    #    try:
    #        assert_almost_equal(estimator.parameters[key], value, decimal=decimal)
    #    except:
    #        raise ValueError('%s predicted:%f, true:%f' % (key, estimator.parameters[key], value))


def test_simulation():

    parameters={
        'B_1A':0.3,
        'B_2A':0.0,
        'B_3A':0.0,
        'C_1A':0.3,
        'C_3A':0.0,
        'C_5A':0.0,
    }

    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 10, 0.1)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    direct_estimator = EstimatorCubic.load(**parameters)

    direct_estimator.is_fitted_=True
    X_pred = direct_estimator.predict(X=X)
    assert_almost_equal(X['phi'].values, X_pred['phi'].values)

def test_fit_simualtion_cheat_p0():

    parameters={
        'B_1A':0.7,
        'B_2A':0.0,
        'B_3A':0.0,
        'C_1A':10.0,
        'C_3A':0.0,
        'C_5A':0.0,
    }

    p0 = parameters

    direct_estimator = EstimatorCubic(fit_method='integration', p0=p0)
    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)

def test_fit_simualtion():

    parameters={
        'B_1A':0.7,
        'B_2A':0.0,
        'B_3A':0.0,
        'C_1A':10.0,
        'C_3A':0.0,
        'C_5A':0.0,
    }

    bounds = {
        'B_1A': (-np.inf, np.inf),  # Assuming only positive coefficients
        'B_2A': (-np.inf, np.inf),  # Assuming only positive coefficients
        'B_3A': (-np.inf, np.inf),  # Assuming only positive coefficients

    }

    direct_estimator = EstimatorCubic(fit_method='integration', bounds=bounds)
    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)

def test_fit_simualtion_quadratic_damping():

    parameters={
        'B_1A':0.7,
        'B_2A':1.5,
        'B_3A':0.0,
        'C_1A':10.0,
        'C_3A':0.0,
        'C_5A':0.0,
    }

    bounds = {
        'B_1A': (-np.inf, np.inf),  # Assuming only positive coefficients
        'B_2A': (-np.inf, np.inf),  # Assuming only positive coefficients
        'B_3A': (-np.inf, np.inf),  # Assuming only positive coefficients

    }

    direct_estimator = EstimatorCubic(fit_method='integration', bounds=bounds)
    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)

def test_fit_simualtion_cubic_damping():

    parameters={
        'B_1A':0.7,
        'B_2A':1.5,
        'B_3A':5.0,
        'C_1A':10.0,
        'C_3A':0.0,
        'C_5A':0.0,
    }

    direct_estimator = EstimatorCubic(fit_method='integration',maxfev=100, )
    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)

def test_fit_simualtion_cubic_damping_not_converged():

    parameters={
        'B_1A':0.7,
        'B_2A':1.5,
        'B_3A':5.0,
        'C_1A':10.0,
        'C_3A':0.0,
        'C_5A':0.0,
    }

    direct_estimator = EstimatorCubic(fit_method='integration',maxfev=1, )
    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)

    with pytest.raises(FitError):
        check(X=X, estimator=direct_estimator, parameters=parameters)

def test_fit_simualtion_quadratic_stiffness():

    parameters={
        'B_1A':0.7,
        'B_2A':0,
        'B_3A':0.0,
        'C_1A':10.0,
        'C_3A':20.0,
        'C_5A':0.0,
    }

    bounds = {
        'B_1A': (-np.inf, np.inf),  # Assuming only positive coefficients
        'B_2A': (-np.inf, np.inf),  # Assuming only positive coefficients
        'B_3A': (-np.inf, np.inf),  # Assuming only positive coefficients

    }

    direct_estimator = EstimatorCubic(fit_method='integration', bounds=bounds)
    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)

def test_fit_simualtion_cubic_stiffness():

    parameters={
        'B_1A':0.7,
        'B_2A':0,
        'B_3A':0.0,
        'C_1A':10.0,
        'C_3A':20.0,
        'C_5A':1000.0,
    }

    bounds = {
        'B_1A': (-np.inf, np.inf),  # Assuming only positive coefficients
        'B_2A': (-np.inf, np.inf),  # Assuming only positive coefficients
        'B_3A': (-np.inf, np.inf),  # Assuming only positive coefficients

    }

    direct_estimator = EstimatorCubic(fit_method='integration',bounds=bounds)
    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)

def test_fit_simualtion_full_cubic():

    parameters={
        'B_1A':0.7,
        'B_2A':1.0,
        'B_3A':3.0,
        'C_1A':10.0,
        'C_3A':10.0,
        'C_5A':0.0,
    }

    direct_estimator = EstimatorCubic(fit_method='integration')
    phi0 = np.deg2rad(20)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)


def test_simulation_quadratic():

    parameters = {
        'B_1A': 100,
        'B_2A': 20,
        'C_1A': 100,

    }

    direct_estimator = EstimatorQuadraticB.load(**parameters)


    phi0 = np.deg2rad(20)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)

    direct_estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0)

def test_fit_simulation_quadratic():

    parameters={
        'B_1A':0.7,
        'B_2A':1.0,
        'B_3A':0.0,
        'C_1A':10.0,
        'C_3A':0.0,
        'C_5A':0.0,
    }

    direct_estimator = EstimatorQuadraticB(fit_method='integration')
    phi0 = np.deg2rad(20)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)

def test_simulation_linear():

    parameters = {
        'B_1A': 100,
        'C_1A': 100,

    }

    direct_estimator = EstimatorLinear.load(**parameters)


    phi0 = np.deg2rad(20)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)

    direct_estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0)

def test_fit_simulation_linear():

    parameters={
        'B_1A':0.7,
        'B_2A':0.0,
        'B_3A':0.0,
        'C_1A':10.0,
        'C_3A':0.0,
        'C_5A':0.0,
    }

    direct_estimator = EstimatorLinear(fit_method='integration')

    phi0 = np.deg2rad(20)
    phi1d0 = 0
    t = np.arange(0, 10, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, parameters=parameters)

def test_result_for_database():

    parameters={
        'B_1A':0.3,
        'B_2A':0.0,
        'B_3A':0.0,
        'C_1A':4.0,
        'C_3A':0.0,
        'C_5A':0.0,
    }

    direct_estimator = EstimatorLinear(fit_method='integration')
    phi0 = np.deg2rad(5)
    phi1d0 = 0
    t = np.arange(0, 20, 0.01)
    X = simulate(t=t, phi0=phi0, phi1d0=phi1d0, **parameters)

    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    direct_estimator.fit(X=X)

    rho=1000
    g=9.81
    meta_data = {
        'GM':1,
        'Volume':1/rho,
        'rho':rho,
        'g':g,
    }

    check(X=X, estimator=direct_estimator, parameters=parameters)
    s = direct_estimator.result_for_database(meta_data=meta_data)

    assert_almost_equal(s['omega0'],np.sqrt(parameters['C_1A']))

    g=9.81
    m=meta_data['Volume']*rho
    C = meta_data['GM']*g*m
    A_44 = C/s['omega0']**2
    assert_almost_equal(s['A_44'],A_44)
    B_1 = s['B_1A']*A_44
    assert_almost_equal(s['B_1'],B_1)
