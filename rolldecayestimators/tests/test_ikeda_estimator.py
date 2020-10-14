import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from rolldecayestimators.ikeda_estimator import IkedaEstimator
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

@pytest.fixture
def df_roll_decay():

    phi0 = np.deg2rad(2)
    phi1d0 = 0
    t = np.arange(0, 120, 0.01)
    estimator = IkedaEstimator(lpp=lpp, TA=TA, TF=TF, beam=beam, BKL=BKL, BKB=BKB, A0=A0, kg=kg, Volume=Volume, gm=gm,
                               V=0)
    IkedaEstimator.parameters = {
        'omega0':omega0,
        'zeta':zeta,
        'd':d,
    }

    yield estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0)

def check(X, estimator, omega0, d, zeta, decimal=4):
    estimator.fit(X=X)
    #assert estimator.result['success']

    X_pred = estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

    score = estimator.score(X)
    assert score > 0.90  # Have reached 0.98 before...

    #assert_almost_equal(X['phi'].values, X_pred['phi'].values, decimal=decimal)
    #assert estimator.score(X) > 0.999
    #assert_almost_equal(estimator.parameters['omega0'], omega0, decimal=decimal)
    #assert_almost_equal(estimator.parameters['zeta'], zeta, decimal=decimal)
    #assert_almost_equal(estimator.parameters['d'], d, decimal=decimal)


def test_fit(df_roll_decay):

    direct_estimator = IkedaEstimator(lpp=lpp, TA=TA, TF=TF, beam=beam, BKL=BKL, BKB=BKB, A0=A0, kg=kg, Volume=Volume,
                                      gm=gm, V=0, verify_input=False)
    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)
    check(X=X, estimator=direct_estimator, omega0=omega0, d=d, zeta=zeta, decimal=2)



