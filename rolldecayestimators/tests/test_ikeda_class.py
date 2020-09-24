import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal

from rolldecayestimators.ikeda import Ikeda


@pytest.fixture
def ikeda():

    N=10
    data = np.zeros(N)
    w=np.linspace(0.1,1, N)
    B_W0 = pd.Series(data=data, index=w)
    fi_a = np.deg2rad(10)
    beam=10
    lpp=100
    kg = 1
    Cb=0.7
    draught = 10
    volume = Cb*lpp*draught*beam
    V = 5
    w = 0.2
    A0 = 0.95

    N_sections = 21
    x_s = np.linspace(0, lpp, N_sections)
    data = {
        'B_s' : beam*np.ones(N_sections),
        'T_s' : draught*np.ones(N_sections),
        'C_s' : A0*np.ones(N_sections),
    }
    sections=pd.DataFrame(data=data, index=x_s)

    i= Ikeda(V=V, draught=draught, w=w, B_W0=B_W0, fi_a=fi_a, beam=beam, lpp=lpp, kg=kg, volume=volume,
             sections=sections)
    i.R=2.0  # Set bilge radius manually

    yield i

def test_R(ikeda):
    assert ikeda.R==2.0

def test_calculate_Ikeda(ikeda):
    B_44=ikeda.calculate_B44()

