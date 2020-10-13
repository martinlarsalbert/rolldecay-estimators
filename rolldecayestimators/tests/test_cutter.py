import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from rolldecayestimators.direct_estimator import DirectEstimator
from rolldecayestimators.transformers import CutTransformer
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

    phi0 = np.deg2rad(10)
    phi1d0 = 0
    t = np.arange(0, 120, 0.01)
    estimator = DirectEstimator()
    estimator.parameters = {
        'omega0':omega0,
        'd':d,
        'zeta':zeta,
    }

    yield estimator.simulate(t=t, phi0=phi0, phi1d0=phi1d0)


def test_cutter(df_roll_decay):

    cut_transformer = CutTransformer(phi_max=np.deg2rad(5))
    cut_transformer.fit(df_roll_decay)
    X = cut_transformer.transform(df_roll_decay)

    fig,ax=plt.subplots()
    df_roll_decay.plot(y='phi', ax=ax, label='raw')
    X.plot(y='phi', ax=ax, label='cut', style='--')
    plt.show()


