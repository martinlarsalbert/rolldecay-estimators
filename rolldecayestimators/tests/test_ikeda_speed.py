import pytest
from rolldecayestimators import ikeda_speed as ikeda
from numpy import sqrt, pi
import numpy as np
from numpy.testing import assert_almost_equal,assert_allclose

visc =   1.15*10**-6  # [m2/s], kinematic viscosity
g    =   9.81
ra   = 1025          # density of water

Cb   =   0.58        # Block coeff
L    =   175         # Length
vcg  =   9.52        # roll axis (vertical centre of gravity) [m]
B    =   25.40       # Breadth of hull [m]
d    =   9.5         # Draught of hull [m]
A    =   0.95*B*d    # Area of cross section of hull [m2]
bBK  =   0.4         # breadth of Bilge keel [m] !!(=height???)
R    =   3           # Bilge Radis


OG = vcg-d           # distance from roll axis to still water level
Ho = B/(2*d)         # half breadth to draft ratio

LBK  = L/4           # Approx
disp = L*B*d*Cb      # Displacement
ND_factor = sqrt(B/(2*g))/(ra*disp*(B**2))   # Nondimensiolizing factor of B44

def test_Bw0():
    Bw0=ikeda.Bw0_S175(w=0.3142)
    assert_allclose(Bw0,1.481175688811950e+06, rtol=0.001)

def test_bw44_V0():
    V = 0  ## Ship speed
    T = 20
    wE = 2 * np.pi * 1 / T  # circular frequency

    bw44 = ikeda.Bw_S175(w=wE, V=V, d=d)

    assert_allclose(bw44, 5.6885e+06, rtol=0.001)

def test_bilge_keel():
    V = 0  ## Ship speed
    T = 20
    wE = 2 * np.pi * 1 / T  # circular frequency
    fi_a = 5 * pi / 180; # roll amplitude !!rad??
    Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0 = ikeda.bilge_keel(wE=wE, fi_a=fi_a, V=V, B=B, d=d, A=A, bBK=bBK, R=R, g=g, OG=OG, Ho=Ho, ra=ra)

    assert_allclose(Bp44BK_N0, 1.384980250651594e+05, rtol=0.001)
    assert_allclose(Bp44BK_H0, 6.137595375551779e+05, rtol=0.001)
    assert_allclose(B44BK_L, 7.221062603618902e+04, rtol=0.001)
    assert_allclose(B44BKW0, 0.014575557482580, rtol=0.001)

def test_friction():
    V = 0  ## Ship speed
    T = 20
    wE = 2 * np.pi * 1 / T  # circular frequency
    fi_a = 5 * pi / 180;  # roll amplitude !!rad??
    B44F = ikeda.frictional(wE=wE, fi_a=fi_a,V=V, B=B, d=d, OG=OG, ra=ra, Cb=Cb, L=L, visc=visc)
    assert_allclose(B44F, 7.015809420450178e+05, rtol=0.001)

def test_hull_lift():
    V = 0  ## Ship speed
    B44L = ikeda.hull_lift(V=V, B=B, d=d, OG=OG, ra=ra, L=L)
    assert_allclose(B44L, 0, rtol=0.001)