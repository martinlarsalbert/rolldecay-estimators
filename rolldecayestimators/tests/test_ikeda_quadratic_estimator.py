import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from rolldecayestimators.ikeda_estimator import IkedaQuadraticEstimator
import matplotlib.pyplot as plt

T0 = 20.0
omega0 = 2.0*np.pi/T0
d=0.0
zeta = 0.044

lpp=100
TA=5
TF=5
beam=10
BKL=0.8*lpp
BKB=0.7
A0=0.95
kg=10
Volume = lpp*beam*TA*0.75
gm=0.5
V = 5

@pytest.fixture
def df_roll_decay():

    phi0 = np.deg2rad(2)
    phi1d0 = 0
    t = np.arange(0, 120, 0.01)
    estimator = IkedaQuadraticEstimator(lpp=lpp, TA=TA, TF=TF, beam=beam, BKL=BKL, BKB=BKB, A0=A0, kg=kg, Volume=Volume, gm=gm, V=0)
    estimator.parameters={
        'omega0':omega0,
        'zeta':zeta,
        'd' : d,
    }

    yield estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0)

def check(X, estimator, omega0, d, zeta, decimal=4):
    estimator.fit(X=X)
    estimator.plot_variation()
    plt.show()
    estimator.plot_B_fit()
    plt.show()

    X_pred = estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

    score = estimator.score(X)
    assert score > 0.90  # Have reached 0.98 before...


def test_fit(df_roll_decay):

    direct_estimator = IkedaQuadraticEstimator(lpp=lpp, TA=TA, TF=TF, beam=beam, BKL=BKL, BKB=BKB, A0=A0, kg=kg,
                                               Volume=Volume, gm=gm, V=0, verify_input=False)
    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, omega0=omega0, d=d, zeta=zeta, decimal=2)

def test_speed(df_roll_decay):

    estimator_zero_speed = IkedaQuadraticEstimator(lpp=lpp, TA=TA, TF=TF, beam=beam, BKL=BKL, BKB=BKB, A0=A0, kg=kg, Volume=Volume, gm=gm, V=0, verify_input=False)
    estimator_zero_speed.fit(X=df_roll_decay)

    estimator_speed = IkedaQuadraticEstimator(lpp=lpp, TA=TA, TF=TF, beam=beam, BKL=BKL, BKB=BKB, A0=A0, kg=kg,
                                                   Volume=Volume, gm=gm, V=V, verify_input=False)
    estimator_speed.fit(X=df_roll_decay)

    fig,ax=plt.subplots()
    estimator_zero_speed.plot_fit(ax=ax)
    estimator_speed.plot_fit(ax=ax, include_model_test=False)

    assert estimator_zero_speed.result['B_44_1'] != estimator_speed.result['B_44_1']
    assert estimator_zero_speed.parameters['zeta']!=estimator_speed.parameters['zeta']
    assert estimator_zero_speed.parameters['d'] != estimator_speed.parameters['d']
    assert estimator_zero_speed.parameters['omega0'] == estimator_speed.parameters['omega0']

    ax.legend()
    plt.show()




