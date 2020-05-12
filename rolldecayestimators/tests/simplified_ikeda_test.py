from rolldecayestimators.simplified_ikeda import calculate_roll_damping, _calculate_roll_damping
import numpy as np
from numpy.testing import assert_almost_equal

"""
L / B = 6.0
B / d = 4.0,
Cb = 0.65,
Cm = 0.98,
φa = 10,
bBK / B = 0.025
lBK / Lpp = 0.2
"""
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

def test_calculate_roll_damping():


    Beam = LPP/L_div_B
    DRAFT = Beam / BD

    lBK = LPP * LBKL
    bBK = Beam * BBKB
    OMEGA = OMEGAHAT/(np.sqrt(Beam / 2 / 9.81))

    OG = DRAFT * OGD

    B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT, BLHAT = calculate_roll_damping(LPP,Beam,CB,CMID,OG,PHI,lBK,bBK,OMEGA,DRAFT)
    assert_almost_equal(B44HAT,0.010156148773301035)


def test_calculate_roll_damping_subfunction():

    Beam = BRTH= LPP/L_div_B
    DRAFT = Beam / BD

    lBK = LPP * LBKL
    bBK = Beam * BBKB
    OMEGA = OMEGAHAT/(np.sqrt(Beam / 2 / 9.81))

    OG = DRAFT * OGD
    TW = 2*np.pi/OMEGA

    B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT, BLHAT = calculate_roll_damping(LPP,Beam,CB,CMID,OG,PHI,lBK,bBK,OMEGA,DRAFT)
    B44HAT2, BFHAT2, BWHAT2, BEHAT2, BBKHAT2 = _calculate_roll_damping(LPP, BRTH, CB, CMID, OGD, PHI, LBKL, BBKB, OMEGA,
                           DRAFT, BD, OMEGAHAT, TW)

    assert B44HAT==B44HAT2
    assert BFHAT == BFHAT2
    assert BWHAT == BWHAT2
    assert BEHAT == BEHAT2
    assert BBKHAT == BBKHAT2

