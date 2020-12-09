import pytest
from rolldecayestimators.simplified_ikeda import calculate_roll_damping, _calculate_roll_damping, limits_kawahara, verify_inputs, SimplifiedIkedaInputError
import rolldecayestimators.ikeda_simple  ## Peter Piehl implementation
import numpy as np
from numpy.testing import assert_almost_equal

"""
L / B = 6.0
B / d = 4.0,
Cb = 0.65,
Cm = 0.98,
φa = 10,
BKB / B = 0.025
BKL / Lpp = 0.2
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
                           DRAFT, BD, OMEGAHAT)

    assert B44HAT==B44HAT2
    assert BFHAT == BFHAT2
    assert BWHAT == BWHAT2
    assert BEHAT == BEHAT2
    assert BBKHAT == BBKHAT2

def test_verify_input():
    Beam = LPP/L_div_B
    DRAFT = Beam / BD

    lBK = LPP * LBKL
    bBK = Beam * BBKB
    OMEGA = OMEGAHAT/(np.sqrt(Beam / 2 / 9.81))
    OG = DRAFT * OGD

    verify_inputs(LPP,Beam,CB,CMID,OG,PHI,lBK,bBK,OMEGA,
                           DRAFT)


def test_verify_input_fail1():
    Beam = LPP/L_div_B
    DRAFT = Beam / BD

    lBK = LPP * LBKL
    bBK = Beam * BBKB
    OMEGA = OMEGAHAT/(np.sqrt(Beam / 2 / 9.81))
    OG = DRAFT * OGD

    CB_fail=0.45
    with pytest.raises(SimplifiedIkedaInputError):

        verify_inputs(LPP,Beam,CB_fail,CMID,OG,PHI,lBK,bBK,OMEGA,
                           DRAFT)

def test_verify_input_fail2():
    Beam = LPP/L_div_B
    DRAFT = Beam / BD

    lBK = LPP * 0.44
    bBK = Beam * BBKB
    OMEGA = OMEGAHAT/(np.sqrt(Beam / 2 / 9.81))
    OG = DRAFT * OGD

    with pytest.raises(SimplifiedIkedaInputError):

        verify_inputs(LPP,Beam,CB,CMID,OG,PHI,lBK,bBK,OMEGA,
                           DRAFT)

def test_limit_input():
    Beam = LPP/L_div_B
    DRAFT = Beam / BD

    lBK = LPP * 0.44
    bBK = Beam * BBKB
    OMEGA = OMEGAHAT/(np.sqrt(Beam / 2 / 9.81))
    OG = DRAFT * OGD

    with pytest.raises(SimplifiedIkedaInputError):

        calculate_roll_damping(LPP,Beam,CB,CMID,OG,PHI,lBK,bBK,OMEGA,DRAFT)

    calculate_roll_damping(LPP, Beam, CB, CMID, OG, PHI, lBK, bBK, OMEGA, DRAFT, limit_inputs=True)

def test_limit_input2():
    Beam = LPP/L_div_B
    DRAFT = Beam / BD

    lBK = LPP * 0.30
    bBK = Beam * 0.001
    OMEGA = OMEGAHAT/(np.sqrt(Beam / 2 / 9.81))
    OG = DRAFT * OGD

    with pytest.raises(SimplifiedIkedaInputError):

        calculate_roll_damping(LPP,Beam,CB,CMID,OG,PHI,lBK,bBK,OMEGA,DRAFT)

    calculate_roll_damping(LPP, Beam, CB, CMID, OG, PHI, lBK, bBK, OMEGA, DRAFT, limit_inputs=True)

def test_limit_input3():
    Beam = LPP/L_div_B
    DRAFT = Beam / BD

    lBK = LPP * 0.30
    bBK = Beam * 0.002
    OMEGA = OMEGAHAT/(np.sqrt(Beam / 2 / 9.81))
    OG = DRAFT*0.25

    with pytest.raises(SimplifiedIkedaInputError):

        calculate_roll_damping(LPP,Beam,CB,CMID,OG,PHI,lBK,bBK,OMEGA,DRAFT)

    calculate_roll_damping(LPP, Beam, CB, CMID, OG, PHI, lBK, bBK, OMEGA, DRAFT, limit_inputs=True)


def random_value(limits):
    return np.random.random()*(limits[1]-limits[0])+limits[0]

def random_values():
    LPP=1.0
    Beam=1.0
    DRAFT=Beam/2.5
    value={
        'LPP':LPP,
        'Beam':Beam,
        'DRAFT':DRAFT,
        'CB':random_value(limits_kawahara['CB']),
        'CMID':random_value(limits_kawahara['CMID']),
        'OG':DRAFT*random_value(limits_kawahara[r'OG/d']),
        'PHI':PHI,
        'BKL':LPP*random_value(limits_kawahara[r'lBk/LPP']),
        'BKB':Beam*random_value(limits_kawahara[r'bBk/B']),
        'OMEGA':random_value(limits_kawahara['OMEGA_hat'])/(np.sqrt(Beam / 2 / 9.81)),

    }
    return value



def test_peter_piehl_implementation():
    np.random.seed(seed=1)

    ikeda = rolldecayestimators.ikeda_simple.Ikeda()  # Peter Piehl

    for i in range(1000):
        r = random_values()
        B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT, BLHAT = calculate_roll_damping(**r)

        # Peter Piehl:

        r2 = {}
        r2['CMID']=r['CMID']
        r2['TW']=(2*np.pi)/r['OMEGA']
        r2['LPP']=r['LPP']
        r2['BRTH']=r['Beam']
        r2['DRAFT']=r['DRAFT']
        r2['CB']=r['CB']
        r2['OG']=r['OG']
        r2['PHI']=r['PHI']
        r2['BKCOMP']=(r['BKL']>0)
        r2['BKL']=r['BKL']
        r2['BKB']=r['BKB']

        ikeda.setPara(r2)
        ikeda.ikedaMethod()

        assert_almost_equal(B44HAT,ikeda.B44HAT)
        assert_almost_equal(BFHAT, ikeda.BFHAT)
        assert_almost_equal(BWHAT, ikeda.BWHAT)
        assert_almost_equal(BEHAT, ikeda.BEHAT)
        assert_almost_equal(BBKHAT, ikeda.BBKHAT)


