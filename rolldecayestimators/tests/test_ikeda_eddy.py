import pytest
import pandas as pd
import numpy as np
from rolldecayestimators import ikeda_naked
from rolldecayestimators import ikeda_speed

from rolldecayestimators.ikeda_speed import calculate_sectional_lewis
from rolldecayestimators import lambdas
from numpy.testing import assert_almost_equal

@pytest.mark.skip('This one does not work yet')
def test_eddy_sections_1():

    B_s = np.array([0.28])
    T_s = 0.112
    sigma = 1.0
    R_b = 0.0
    g = 9.81
    OG = 0.0
    L = 0.8
    volume = 0.02509
    rho=1000.0

    S_s = sigma * B_s * T_s
    a, a_1, a_3, sigma_s, H0 = calculate_sectional_lewis(B_s=B_s, T_s=T_s, S_s=S_s)
    assert H0==1.25

    w_hat = 1.0
    phi_a = 0.175
    w = lambdas.omega_from_hat(beam=B_s, g=g, omega_hat=w_hat)

    #R_b = ikeda_eddy.calculate_R_b(beam=B_s, draught=T_s, H0=H0, sigma=sigma_s)

    B_E0_s = ikeda_speed.eddy_sections(bwl=B_s, a_1=a_1, a_3=a_3, sigma=sigma_s, H0=H0, Ts=T_s, OG=OG, R=R_b, wE=w,
                                       fi_a=phi_a, ra=rho)

    B_E0 = B_E0_s * L

    Disp = volume
    B_E0_hat = lambdas.B_to_hat_lambda(B=B_E0, Disp=Disp, beam=B_s, g=g, rho=rho)
    B_E0_star_hat = B_E0_hat * 3 * np.pi / 8

    assert_almost_equal(B_E0_star_hat, 0.042)

@pytest.mark.skip('This one does not work yet')
def test_eddy_sections_2():

    B_s = 0.28
    T_s = 0.112
    sigma = 1.0
    R_b = 0.0
    g = 9.81
    OG = 0.0
    L = 0.8
    volume = 0.02509
    rho=1000.0

    S_s = sigma * B_s * T_s
    a, a_1, a_3, sigma_s, H0 = calculate_sectional_lewis(B_s=B_s, T_s=T_s, S_s=S_s)
    assert H0==1.25

    w_hat = 1.0
    phi_a = 0.175
    w = lambdas.omega_from_hat(beam=B_s, g=g, omega_hat=w_hat)

    B_E0_s = ikeda_naked.eddy_sections(bwl=B_s, a_1=a_1, a_3=a_3, sigma=sigma_s, H0=H0, Ts=T_s,
                  OG=OG, R=R_b, wE=w, fi_a=phi_a, ra=rho)

    B_E0 = B_E0_s * L

    Disp = volume
    B_E0_hat = lambdas.B_to_hat_lambda(B=B_E0, Disp=Disp, beam=B_s, g=g, rho=rho)
    B_E0_star_hat = B_E0_hat * 3 * np.pi / 8

    assert_almost_equal(B_E0_star_hat, 0.042)

def test_calculate_Cr():

    df_kvlcc2 = pd.Series({
        'area' :    917.895066,
        'x' :   61.597568,
        'd':    20.800000,
        'B':    56.159232,
        'R':    34.146143,
        'sigma':    0.785794,
        'OG/d':    0.000000,
        'a_1':    0.148893,
        'a_3': - 0.000246,
        'H0':    1.349982,
    })

    OG = df_kvlcc2['OG/d']*df_kvlcc2.d
    ra = 1000
    df_kvlcc2 = pd.DataFrame(df_kvlcc2).transpose()

    C_r = ikeda_naked.calculate_C_r(bwl=df_kvlcc2.B,
                          a_1=df_kvlcc2.a_1, a_3=df_kvlcc2.a_3, sigma=df_kvlcc2.sigma, H0=df_kvlcc2.H0, d=df_kvlcc2.d,
                          OG=OG,
                          R=df_kvlcc2.R, ra=ra)