import numpy as np
import pytest
from rolldecayestimators import tests
import rolldecayestimators
import pandas as pd
import os
from rolldecayestimators import ikeda_speed as ikeda
from numpy import pi, sqrt
from numpy.testing import assert_almost_equal
from matplotlib import pyplot as plt

@pytest.fixture
def lewis_coefficients():
    file_path = os.path.join(rolldecayestimators.path,'faust_lewis.csv')
    df = pd.read_csv(file_path, sep=';')

    yield df

def test_eddy():

    N = 21
    lpp=100.0
    OG=5.5
    R=2
    d=1
    wE=0.2
    fi_a=np.deg2rad(3)
    xs = np.linspace(0,lpp, N)
    B = T = S = np.ones(N)
    a, a_1, a_3, sigma_s, H = ikeda.calculate_sectional_lewis(B_s=B, T_s=T, S_s=S)

    B_E = ikeda.eddy(bwl=B, a_1=a_1, a_3=a_3, sigma=sigma_s, xs=xs, H0=H, Ts=T, OG=OG, R=R, wE=wE, fi_a=fi_a)

def test_eddy_faust(lewis_coefficients):
    """
    Reproduction of Carl-Johans Matlab implementation for Faust.
    Parameters
    ----------
    lewis_coefficients

    Returns
    -------

    """

    lc=lewis_coefficients

    T = 27.6
    wE = 2*pi*1/T  #  circular frequency
    d    =   9.5  # Draught of hull [m]
    vcg  =   14.9  # roll axis (vertical centre of gravity) [m]
    OG = -1 * (vcg - d)   # distance from roll axis to still water level
    fi_a =   10*pi/180  # roll amplitude !!rad??
    R = 5  # Bilge Radis
    B_E = ikeda.eddy(bwl=lc['bwl'], a_1=lc['a1'], a_3=lc['a3'], sigma=lc['sigma'], xs=lc['x'], H0=lc['H'], Ts=lc['Ts'],
                     OG=OG, R=R, wE=wE, fi_a=fi_a)

    #assert_almost_equal(actual=B_E, desired=1175062.2691943)

    ScaleF =  1#/29.565                  # Scale Factor [-]
    Cb   =   0.61                        # Block coeff
    L    =   220*ScaleF                  # Length
    vcg  =   14.9*ScaleF                 # roll axis (vertical centre of gravity) [m]
    B    =   32.26*ScaleF                # Breadth of hull [m]
    d    =   9.5*ScaleF                  # Draught of hull [m]
    g    =   9.81

    ra   = 1025                   # density of water

    disp = L*B*d*Cb  # Displacement
    ND_factorB = sqrt(B / (2 * g)) / (ra * disp * (B**2))

    w_hat = np.linspace(0,1,100)
    w = sqrt(2) * w_hat / sqrt(B / g)
    B_E = ikeda.eddy(bwl=lc['bwl'], a_1=lc['a1'], a_3=lc['a3'], sigma=lc['sigma'], xs=lc['x'], H0=lc['H'], Ts=lc['Ts'],
                     OG=OG, R=R, wE=w, fi_a=fi_a)

    B_E_hat = B_E*ND_factorB
    fig,ax=plt.subplots()
    ax.plot(w_hat, B_E_hat)
    plt.show()



def test_calculate_sectional_lewis():
    N=21
    B=T=S=np.ones(N)
    a, a_1, a_3, sigma_s, H = ikeda.calculate_sectional_lewis(B_s=B, T_s=T, S_s=S)