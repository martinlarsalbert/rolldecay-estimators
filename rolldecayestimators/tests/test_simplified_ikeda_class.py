import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal_nulp

from rolldecayestimators.simplified_ikeda_class import SimplifiedIkeda
from rolldecayestimators.simplified_ikeda import calculate_roll_damping

L_div_B = 6.0
BD = 4.0
CB = 0.65
CMID = 0.98
PHI = 10
BBKB = 0.025
LBKL = 0.2
OGD = -0.2
OMEGAHAT = 0.6
LPP = 300

Beam = LPP / L_div_B
DRAFT = Beam / BD

lBK = LPP * LBKL
bBK = Beam * BBKB
OMEGA = OMEGAHAT / (np.sqrt(Beam / 2 / 9.81))

OG = DRAFT * OGD
kg = DRAFT - OG
volume = LPP * DRAFT * Beam * CB

B44HAT1, BFHAT1, BWHAT1, BEHAT1, BBKHAT1, BLHAT1 = calculate_roll_damping(LPP, Beam, CB, CMID, OG, PHI, lBK, bBK, OMEGA,
                                                                          DRAFT)
decimal=5

def test_simplified_ikeda():


    si = SimplifiedIkeda(V=0, w=OMEGA, fi_a=np.deg2rad(PHI), beam=Beam, lpp=LPP, kg=kg, volume=volume, draught=DRAFT, A0=CMID,
                    lBK=lBK, bBK=bBK, visc=1.14e-6)

    B44HAT = si.calculate_B44()
    BFHAT = si.calculate_B_F()
    BWHAT = si.calculate_B_W()
    BBKHAT = si.calculate_B_BK()
    BLHAT = si.calculate_B_L()


    assert_almost_equal(BFHAT, BFHAT1, decimal=decimal)
    assert_almost_equal(BWHAT, BWHAT1, decimal=decimal)
    assert_almost_equal(BBKHAT, BBKHAT1, decimal=decimal)
    assert_almost_equal(BLHAT, BLHAT1, decimal=decimal)
    assert_almost_equal(B44HAT, B44HAT1, decimal=decimal)

def test_simplified_ikeda_vector():

    N=10
    w = np.ones(N)*OMEGA
    si = SimplifiedIkeda(V=0, w=w, fi_a=np.deg2rad(PHI), beam=Beam, lpp=LPP, kg=kg, volume=volume, draught=DRAFT, A0=CMID,
                    lBK=lBK, bBK=bBK, visc=1.14e-6)

    B44HAT = si.calculate_B44()

    assert_almost_equal(B44HAT, np.ones(N)*B44HAT1, decimal=decimal)

def test_simplified_ikeda_vector2():

    N=10
    fi_a = np.deg2rad(np.linspace(1, 15, N))

    si = SimplifiedIkeda(V=0, w=OMEGA, fi_a=fi_a, beam=Beam, lpp=LPP, kg=kg, volume=volume, draught=DRAFT, A0=CMID,
                    lBK=lBK, bBK=bBK, visc=1.14e-6)

    B44HAT = si.calculate_B44()

    a = 1
