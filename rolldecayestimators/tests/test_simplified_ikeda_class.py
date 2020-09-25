import numpy as np
from numpy.testing import assert_almost_equal

from rolldecayestimators.simplified_ikeda_class import SimplifiedIkeda
from rolldecayestimators.simplified_ikeda import calculate_roll_damping

def test_simplified_ikeda():

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
    kg = DRAFT-OG
    volume=LPP*DRAFT*Beam*CB

    B44HAT1, BFHAT1, BWHAT1, BEHAT1, BBKHAT1, BLHAT1 = calculate_roll_damping(LPP, Beam, CB, CMID, OG, PHI, lBK, bBK, OMEGA,
                                                                        DRAFT)


    si = SimplifiedIkeda(V=0, w=OMEGA, fi_a=np.deg2rad(PHI), beam=Beam, lpp=LPP, kg=kg, volume=volume, draught=DRAFT, A0=CMID,
                    lBK=lBK, bBK=bBK, visc=1.14e-6)

    B44HAT = si.calculate_B44()
    BFHAT = si.calculate_B_F()
    BWHAT = si.calculate_B_W()
    BBKHAT = si.calculate_B_BK()
    BLHAT = si.calculate_B_L()

    decimal=5
    assert_almost_equal(BFHAT, BFHAT1, decimal=decimal)
    assert_almost_equal(BWHAT, BWHAT1, decimal=decimal)
    assert_almost_equal(BBKHAT, BBKHAT1, decimal=decimal)
    assert_almost_equal(BLHAT, BLHAT1, decimal=decimal)
    assert_almost_equal(B44HAT, B44HAT1, decimal=decimal)
