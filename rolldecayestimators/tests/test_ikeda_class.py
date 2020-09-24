import pytest
import pandas as pd
import numpy as np
from numpy import pi, sqrt
import os
from numpy.testing import assert_almost_equal, assert_allclose

from rolldecayestimators.ikeda import Ikeda
import rolldecayestimators


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

@pytest.fixture
def ikeda_faust():
    # this is indata from Carl-Johans matlab example for ship: Faust.

    ScaleF = 1  # %/29.565;                # Scale Factor [-]
    visc = 1.15 * 10 ** -6;  # [m2/s], kinematic viscosity
    Cb = 0.61;  # Block coeff
    L = 220 * ScaleF;  # Length
    vcg = 14.4 * ScaleF;  # roll axis (vertical centre of gravity) [m]
    vcg = 14.9 * ScaleF;  # roll axis (vertical centre of gravity) [m]
    B = 32.26 * ScaleF;  # Breadth of hull [m]
    d = 9.5 * ScaleF;  # Draught of hull [m]
    A = 0.93 * B * d;  # Area of cross section of hull [m2]
    bBK = 0.4 * ScaleF;  # breadth of Bilge keel [m] !!(=height???)
    R = 5 * ScaleF;  # Bilge Radis
    g = 9.81;
    C_mid = 0.93;

    OG = -1 * (vcg - d)  # *0.8;                    # distance from roll axis to still water level
    Ho = B / (2 * d);  # half breadth to draft ratio
    ra = 1025;  # density of water

    # locals
    LBK = L / 4;  # Approx
    disp = L * B * d * Cb;  # Displacement

    # variables!!
    T = 27.6 * sqrt(ScaleF);
    wE = 2 * pi * 1 / T;  # circular frequency
    fi_a = 10 * pi / 180;  # roll amplitude !!rad??
    V = 0;  # Speed

    data_path_faust = os.path.join(rolldecayestimators.path, 'Bw0_faust.csv')
    data_faust = pd.read_csv(data_path_faust, sep=';')
    data_faust.set_index('w_vec', inplace=True)
    B_W0 = data_faust['b44_vec']

    N_sections = 21
    x_s = np.linspace(0, L, N_sections)
    data = {
        'B_s': B * np.ones(N_sections),
        'T_s': d * np.ones(N_sections),
        'C_s': C_mid*np.ones(N_sections),
    }
    sections = pd.DataFrame(data=data, index=x_s)  # Fake sections (not testing the eddy)

    i = Ikeda(V=V, draught=d, w=wE, B_W0=B_W0, fi_a=fi_a, beam=B, lpp=L, kg=vcg, volume=disp,
              sections=sections)
    i.R = R  # Set bilge radius manually

    yield i

def test_R(ikeda):
    assert ikeda.R==2.0

def test_calculate_Ikeda(ikeda):
    B_44=ikeda.calculate_B44()

def test_calculate_Ikeda_faust(ikeda_faust):
    B_44=ikeda_faust.calculate_B44()

def test_Bw0(ikeda_faust):
    Bw0=ikeda_faust.calculate_B_W0()
    assert_allclose(Bw0, 1895860.700098, rtol=0.001)


@pytest.mark.skip('Not ready...')
def test_bw44_V0(ikeda_faust):

    ikeda_faust.V = 0  ## Ship speed
    bw44 = ikeda_faust.calculate_B_W()
    assert_allclose(bw44, 1895860.700098, rtol=0.001)

"""
def test_bilge_keel():
    V = 0  ## Ship speed
    T = 27.6 * sqrt(ScaleF);
    wE = 2 * pi * 1 / T;  # circular frequency
    fi_a = 10 * pi / 180;  # roll amplitude !!rad??
    Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0 = ikeda.bilge_keel(w=wE, fi_a=fi_a, V=V, B=B, d=d, A=A, bBK=bBK, R=R, g=g, OG=OG, Ho=Ho, ra=ra)

    assert_allclose(Bp44BK_N0, 3.509033728796537e+05, rtol=0.001)
    assert_allclose(Bp44BK_H0, 1.057765842625329e+06, rtol=0.001)
    assert_allclose(B44BK_L, 2.607161328890767e+05, rtol=0.001)
    assert_allclose(B44BKW0, 0.011960200569156, rtol=0.001)

def test_friction():
    V = 0  ## Ship speed
    T = 27.6 * sqrt(ScaleF);
    wE = 2 * pi * 1 / T;  # circular frequency
    fi_a = 10 * pi / 180;  # roll amplitude !!rad??
    B44F = ikeda.frictional(w=wE, fi_a=fi_a, V=V, B=B, d=d, OG=OG, ra=ra, Cb=Cb, L=L, visc=visc)
    assert_allclose(B44F, 5.794039754129194e+06, rtol=0.001)

def test_hull_lift():
    V = 10  ## Ship speed
    B44L = ikeda.hull_lift(V=V, B=B, d=d, OG=OG, ra=ra, L=L, A=A)
    assert_allclose(B44L, 1.734463413980598e+08, rtol=0.001)

"""