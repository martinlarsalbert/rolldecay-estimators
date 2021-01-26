import pytest
import pandas as pd
import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt
import os
from numpy.testing import assert_almost_equal, assert_allclose

from rolldecayestimators.ikeda import Ikeda, IkedaR
from rolldecayestimators import lambdas
import rolldecayestimators
import pyscores2.test
import pyscores2.indata
import pyscores2.output

@pytest.fixture
def ikeda():

    N=10
    data = np.zeros(N)
    w_hat=np.linspace(0.1,1, N)
    B_W0_hat = pd.Series(data=data, index=w_hat)
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

    i= Ikeda(V=V, w=w, B_W0_hat=B_W0_hat, fi_a=fi_a, beam=beam, lpp=lpp, kg=kg, volume=volume,
             sections=sections, BKL=0, BKB=0)
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
    data_faust['w_hat'] = lambdas.omega_hat(beam=B, g=g, omega0=data_faust['w_vec'])
    data_faust['B_W0_hat'] = lambdas.B_hat_lambda(B=data_faust['b44_vec'], Disp=disp, beam=B, g=g, rho=ra)

    data_faust.set_index('w_hat', inplace=True)
    B_W0_hat = data_faust['B_W0_hat']


    N_sections = 21
    x_s = np.linspace(0, L, N_sections)
    data = {
        'B_s': B * np.ones(N_sections),
        'T_s': d * np.ones(N_sections),
        'C_s': C_mid*np.ones(N_sections),
    }
    sections = pd.DataFrame(data=data, index=x_s)  # Fake sections (not testing the eddy)

    i = Ikeda(V=V, w=wE, B_W0_hat=B_W0_hat, fi_a=fi_a, beam=B, lpp=L, kg=vcg, volume=disp,
              sections=sections, BKB=bBK, BKL=LBK)
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
    assert_allclose(Bw0, 5.541101e-05, rtol=0.001)


def test_bw44_V0(ikeda_faust):

    ikeda_faust.V = 0  ## Ship speed
    bw44 = ikeda_faust.calculate_B_W()
    assert_allclose(bw44, 5.541101e-05, rtol=0.01)


def test_bilge_keel(ikeda_faust):

    ikeda_faust.V = 0  ## Ship speed
    T = 27.6
    ikeda_faust.w = 2 * pi * 1 / T;  # circular frequency
    ikeda_faust.fi_a = 10 * pi / 180;  # roll amplitude !!rad??

    B_BK = ikeda_faust.calculate_B_BK()

    assert_allclose(B_BK, ikeda_faust.B_hat(75841485), rtol=0.01)

@pytest.mark.skip('This one does not work yet')
def test_friction(ikeda_faust):
    ikeda_faust.V = 0  ## Ship speed
    T = 27.6
    ikeda_faust.w = 2 * pi * 1 / T;  # circular frequency
    ikeda_faust.fi_a = 10 * pi / 180;  # roll amplitude !!rad??

    B44F = ikeda_faust.calculate_B_F()

    assert_allclose(B44F, ikeda_faust.B_hat(5652721), rtol=0.001)


def test_hull_lift(ikeda_faust):

    ikeda_faust.V = 10  ## Ship speed
    B44L = ikeda_faust.calculate_B_L()
    assert_allclose(B44L, ikeda_faust.B_hat(1.692159e+08), rtol=0.001)

@pytest.fixture
def indata():
    indata=pyscores2.indata.Indata()
    indata.open(indataPath=pyscores2.test.indata_path)
    yield indata

@pytest.fixture
def output():
    output=pyscores2.output.OutputFile(filePath=pyscores2.test.outdata_path)
    yield output

def test_load_scoresII(indata, output):

    V = 5
    w = 0.2
    fi_a = np.deg2rad(10)

    ikeda = Ikeda.load_scoresII(indata=indata, output_file=output, V=V, w=w, fi_a=fi_a, BKB=0, BKL=0, kg=0)
    ikeda.R = 2.0  # Set bilge radius manually
    B_44_hat = ikeda.calculate_B44()

def test_calculate_R_b(indata, output):

    V = 5
    w = 0.2
    fi_a = np.deg2rad(10)

    ikeda = Ikeda.load_scoresII(indata=indata, output_file=output, V=V, w=w, fi_a=fi_a, BKB=0, BKL=0, kg=0)
    R_b = ikeda.calculate_R_b()


def test_load_scoresII_scale(indata, output):

    V = 5
    w = 0.2
    fi_a = np.deg2rad(10)
    R=2.0

    ikeda = Ikeda.load_scoresII(indata=indata, output_file=output, V=V, w=w, fi_a=fi_a, BKB=0, BKL=0, kg=0)
    ikeda.R = R  # Set bilge radius manually

    scale_factor=50
    V_m=V/np.sqrt(scale_factor)
    w_m=w*np.sqrt(scale_factor)

    ikeda_model = Ikeda.load_scoresII(indata=indata, output_file=output, V=V_m, w=w_m, fi_a=fi_a, BKB=0, BKL=0,
                                      scale_factor=scale_factor, kg=0)
    ikeda_model.R = R/scale_factor  # Set bilge radius manually

    assert_almost_equal(ikeda.calculate_B_W(),  ikeda_model.calculate_B_W() )
    assert_almost_equal(ikeda.calculate_B_BK(), ikeda_model.calculate_B_BK())
    assert_almost_equal(ikeda.calculate_B_E(),  ikeda_model.calculate_B_E() )
    assert_almost_equal(ikeda.calculate_B_L(),  ikeda_model.calculate_B_L() )

def test_load_scoresII_scale_V_variation(indata, output):

    scale_factor = 68
    N = 200
    V = np.linspace(0, 15.5, N) * 1.852 / 3.6 / np.sqrt(scale_factor)
    kg = 0.2735294117647059
    w = 2.4755750032144674

    ## Load ScoresII results
    phi_a_deg = 10
    phi_a = np.deg2rad(phi_a_deg) * np.ones(N)

    ikeda_estimator = Ikeda.load_scoresII(V=V,
                                          w=w,
                                          fi_a=phi_a,
                                          indata=indata,
                                          output_file=output,
                                          scale_factor=scale_factor, BKL=0, BKB=0, kg=kg)

    results = ikeda_estimator.calculate()
    results['V'] = V
    results.set_index('V', inplace=True)

    phi_as = np.deg2rad(np.linspace(0, phi_a_deg, N))
    ikeda_estimator2 = Ikeda.load_scoresII(V=np.max(V),
                                           w=w,
                                           fi_a=phi_as,
                                           indata=indata,
                                           output_file=output,
                                           scale_factor=scale_factor, BKL=0, BKB=0, kg=kg)

    results2 = ikeda_estimator2.calculate()
    results2['phi_a'] = phi_as
    results2.set_index('phi_a', inplace=True)

    fig,axes=plt.subplots(ncols=2)

    ax=axes[0]
    results.plot(y='B_44_hat',ax=ax)

    ax = axes[1]
    results2.plot(y='B_44_hat', ax=ax)

    fig.show()

    assert_almost_equal(results.iloc[-1]['B_44_hat'], results2.iloc[-1]['B_44_hat'])
