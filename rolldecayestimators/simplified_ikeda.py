"""
Python conversion of fortran code by Martin Alexandersson
************************************************************************
*  Simple Prediction Formula of Roll Damping                           *
*                         on the Basis of Ikeda's Method               *
*                                                                      *
*  Roll_Damping.for            coded by Yoshiho IKEDA                  *
*                                       Yuki KAWAHARA                  *
*                                       Kazuya MAEKAWA                 *
*                                       Osaka Prefecture University    *
*                                       Graduate School of Engineering *
*  last up date 2009.07.08                                             *
************************************************************************
"""

from numpy import exp as EXP
from numpy import log as LOG
from numpy import sqrt as SQRT
from numpy import pi as PI
import numpy as np
import pandas as pd
from rolldecayestimators import ikeda_speed

def calculate_roll_damping(LPP, Beam, CB, CMID, OG, PHI, BKL, BKB, OMEGA,
                           DRAFT, V=0, KVC = 1.14e-6, verify_input=True, limit_inputs=False, Bw_div_Bw0_max=np.inf,
                           BWHAT_lim=np.inf, rho=1000, alternative_bilge_keel=False, RdivB=0.02):
    """
    ********************************************************************
    *** Calculation of roll damping by the proposed predition method ***
    ********************************************************************

    :param LPP: [m]
    :param Beam: [m]
    :param CB: Block coefficient [-]
    :param CMID: Middship coefficient (A_m/(B*d) [-]
    :param OG: distance from the still water level O to the roll axis G [m]
    :param PHI: Roll angle [deg]
    :param BKL: length of bilge keel [m]
    :param BKB: height of bilge keel [m]
    :param OMEGA: Frequency of motion [rad/s]
    :param DRAFT: DRAFT : ship draught [m]
    :param OMEGAHAT:
    :param V: ship speed [m/s]
    :param KVC = 1.14e-6  # Kinematic Viscosity Coefficient
    :param verify_input = True, should the inputs be verified to be within limits?
    :param limit_inputs = False, use limit value if input limit is exceeded.
    :param Bw_div_Bw0_max=12, There are some problems with the wave damping speed dependence being over predicted this
        is a limit. Set it to np.inf to turn it off.
    :param BWHAT_lim=0.005, There are some problems with the wave damping being over predicted this
        is a limit. Set it to np.inf to turn it off.
    :param rho, water density
    :param alternative_bilge_keel, if True an alternative bilge keel calculation is used.
    :RdivB, bilge radius / ship beam (only used if alternative_bilge_keel=True)

    :return: B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT
     Nondimensional damping:
    B44HAT: Total
    BFHAT: Friction
    BWHAT: Wave
    BEHAT: Eddy
    BBKHAT: Bilge keel
    """

    # Check if any input is null:
    inputs = {
        'LPP': LPP,
        'Beam': Beam,
        'CB': CB,
        'CMID': CMID,
        'OG': OG,
        'PHI': PHI,
        'BKL': BKL,
        'BKB': BKB,
        'OMEGA': OMEGA,
        'DRAFT': DRAFT,
    }
    inputs=pd.Series(inputs)
    mask = pd.isnull(inputs)
    nulls = inputs[mask]
    if len(nulls)>0:
        raise SimplifiedIkedaInputError('%s is NaN' % nulls)

    if limit_inputs:
        # Limit the inputs to not exceed the limits.
        outputs = _limit_inputs(LPP=LPP, Beam=Beam, CB=CB, CMID=CMID, OG=OG, PHI=PHI, lBK=BKL, bBK=BKB, OMEGA=OMEGA, DRAFT=DRAFT)
        LPP = outputs['LPP']
        Beam = outputs['Beam']
        CB = outputs['CB']
        CMID = outputs['CMID']
        OG = outputs['OG']
        PHI = outputs['PHI']
        BKL = outputs['BKL']
        BKB = outputs['BKB']
        OMEGA = outputs['OMEGA']
        DRAFT = outputs['DRAFT']

    if verify_input:
        verify_inputs(LPP, Beam, CB, CMID, OG, PHI, BKL, BKB, OMEGA,
                      DRAFT)

    LBKL= BKL / LPP
    BD = Beam/DRAFT
    OGD = OG/DRAFT
    BBKB = BKB / Beam
    BRTH = Beam
    #OMEGA/= 2  # Magic factor!??? This value seemt o give better results..?
    OMEGAHAT = OMEGA * SQRT(BRTH / 2 / 9.81)

    B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT = _calculate_roll_damping(LPP=LPP,BRTH=BRTH, CB=CB, CMID=CMID, OGD=OGD, PHI=PHI, LBKL=LBKL, BBKB=BBKB,
                                   OMEGA=OMEGA, DRAFT=DRAFT, BD=BD, OMEGAHAT=OMEGAHAT, KVC=KVC)

    # Speed dependance:
    # Hull lift:
    A=CMID*Beam*DRAFT
    BL=ikeda_speed.hull_lift(V=V,B=Beam, d=DRAFT, OG=OG, L=LPP, A=A, ra=1025)
    disp=LPP*Beam*DRAFT*CB
    ND_factorB = np.sqrt(Beam / (2 * 9.81)) / (1025 * disp * (Beam**2));  # Nondimensiolizing factor of B44
    BLHAT=BL*ND_factorB
    B44HAT+=BLHAT

    # Wave speed dependance:
    if V>0:
        BWHAT_speed =ikeda_speed.Bw(w=OMEGA, V=V, d=DRAFT, Bw0=BWHAT, Bw_div_Bw0_max=Bw_div_Bw0_max)

        if BWHAT_speed>BWHAT_lim:
            BWHAT_speed=BWHAT_lim # This limit give some improvement

        B44HAT = B44HAT - BWHAT + BWHAT_speed
        BWHAT = BWHAT_speed

    # Eddy speed dependence:
    if V>0:
        factor=(0.04*OMEGA*LPP/V)**2
        B44HAT-=BEHAT
        BEHAT*=(factor)/(1+factor)
        B44HAT+=BEHAT

    if alternative_bilge_keel and BKB>0:

        B44HAT-=BBKHAT

        A=CMID*Beam*DRAFT
        g=9.81
        Ho = Beam / (2 * DRAFT)  # half breadth to draft ratio
        R = RdivB*Beam
        Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0 = ikeda_speed.bilge_keel(w=OMEGA, fi_a=np.deg2rad(PHI), V=V, B=Beam, d=DRAFT, A=A, bBK=BKB, R=R, g=g, OG=OG, Ho=Ho, ra=rho)
        B44BK_N0 = Bp44BK_N0 * BKL
        B44BK_H0 = Bp44BK_H0 * BKL
        B44BK_L = B44BK_L
        # B44BKW0 = B44BKW0 * dim...
        B44_BK = B44BK_N0 + B44BK_H0 + B44BK_L
        BBKHAT=ND_factorB*B44_BK
        B44HAT+=BBKHAT

    return B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT, BLHAT

def _calculate_roll_damping(LPP, BRTH, CB, CMID, OGD, PHI, LBKL, BBKB, OMEGA,
                           DRAFT, BD, OMEGAHAT, KVC = 1.14e-6):
    """
    ********************************************************************
    *** Calculation of roll damping by the proposed predition method ***
    ********************************************************************

    :param LPP: [m]
    :param Beam: [m]
    :param CB: Block coefficient [-]
    :param CMID: Middship coefficient (A_m/(B*d) [-]
    :param OGD: OG/DRAFT
    :param PHI: Roll angle [deg]
    :param LBKL: BKL/LPP
    :param BBKB : BKB/Beam
    :param OMEGA: Frequency of motion [rad/s]
    :param DRAFT: DRAFT : ship draught [m]
    :param BD: Beam/DRAFT
    :param OMEGAHAT:
    :param KVC = 1.14e-6  # Kinematic Viscosity Coefficient

    :return: B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT
     Nondimensional damping:
    B44HAT: Total
    BFHAT: Friction
    BWHAT: Wave
    BEHAT: Eddy
    BBKHAT: Bilge keel
    """

    # *** Frictional Component ***
    BFHAT = calculate_B_F(BD, BRTH, CB, DRAFT, KVC, LPP, OGD, OMEGA, PHI)

    #*** Wave Component ***
    BWHAT = calculate_B_W0(BD, CB, CMID, OGD, OMEGAHAT)

    #*** Eddy Component ***
    BEHAT = calculate_B_E(BD, CB, CMID, OGD, OMEGAHAT, PHI)

    #*** Bilge Keel Component ***
    BBKHAT = calculate_B_BK(BBKB, BD, CB, CMID, LBKL, OGD, OMEGAHAT, PHI)

    #*** Total Roll Damping ***
    B44HAT=BFHAT+BWHAT+BEHAT+BBKHAT

    return B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT


def calculate_B_BK(BBKB, BD, CB, CMID, LBKL, OGD, OMEGAHAT, PHI):
    """
    Bilge keel damping

    :param BBKB : BKB/Beam
    :param BD: Beam/DRAFT
    :param CB: Block coefficient [-]
    :param CMID: Middship coefficient (A_m/(B*d) [-]
    :param LBKL: BKL/LPP
    :param OGD: OG/DRAFT
    :param OMEGAHAT:
    :param PHI: Roll angle [deg]
    :return: BBKHAT
    """


    FBK1 = (-0.3651 * CB + 0.3907) * (BD - 2.83) ** 2 - 2.21 * CB + 2.632
    FBK2 = 0.00255 * PHI ** 2 + 0.122 * PHI + 0.4794
    FBK3 = (-0.8913 * BBKB ** 2 - 0.0733 * BBKB) * LBKL ** 2 + (
                5.2857 * BBKB ** 2 - 0.01185 * BBKB + 0.00189) * LBKL
    ABK = FBK1 * FBK2 * FBK3
    BBK1 = (5.0 * BBKB + 0.3 * BD - 0.2 * LBKL + 0.00125 * PHI ** 2 - 0.0425 * PHI - 1.86) * OGD
    BBK2 = -15.0 * BBKB + 1.2 * CB - 0.1 * BD - 0.0657 * OGD ** 2 + 0.0586 * OGD + 1.6164
    BBK3 = 2.5 * OGD + 15.75
    BBKHAT = ABK * EXP(BBK1 + BBK2 * CMID ** BBK3) * OMEGAHAT

    LBKL = np.array(LBKL)
    BBKHAT = np.array(BBKHAT)

    mask = LBKL == 0
    BBKHAT[mask]=0

    return BBKHAT


def calculate_B_E(BD, CB, CMID, OGD, OMEGAHAT, PHI):
    """
    Eddy damping B_E

    :param BD: Beam/DRAFT
    :param CB: Block coefficient [-]
    :param CMID: Middship coefficient (A_m/(B*d) [-]
    :param OGD: OG/DRAFT
    :param OMEGAHAT:
    :param PHI: Roll angle [deg]

    :return: BEHAT
    """

    FE1 = (-0.0182 * CB + 0.0155) * (BD - 1.8) ** 3
    FE2 = -79.414 * CB ** 4 + 215.695 * CB ** 3 - 215.883 * CB ** 2 + 93.894 * CB - 14.848
    AE = FE1 + FE2
    BE1 = (3.98 * CB - 5.1525) * (-0.2 * BD + 1.6) * OGD * (
                (0.9717 * CB ** 2 - 1.55 * CB + 0.723) * OGD + 0.04567 * CB + 0.9408)
    BE2 = (0.25 * OGD + 0.95) * OGD - 219.2 * CB ** 3 + 443.7 * CB ** 2 - 283.3 * CB + 59.6
    BE3 = -15 * CB * BD + 46.5 * CB + 11.2 * BD - 28.6
    CR = AE * EXP(BE1 + BE2 * CMID ** BE3)
    BEHAT = 4.0 * OMEGAHAT * PHI * PI / 180 / (3.0 * PI * CB * BD ** 3.0) * CR
    return BEHAT


def calculate_B_W0(BD, CB, CMID, OGD, OMEGAHAT):
    """
    Wave roll damping at zero speed B_W0

    :param BD: Beam/DRAFT
    :param CB: Block coefficient [-]
    :param CMID: Middship coefficient (A_m/(B*d) [-]
    :param OGD: OG/DRAFT
    :param OMEGAHAT:

    :return:B_W0

    """

    X1 = BD;
    X2 = CB;
    X3 = CMID
    X5 = OMEGAHAT
    X4 = 1 - OGD
    A111 = -0.002222 * X1 ** 3 + 0.040871 * X1 ** 2 - 0.286866 * X1 + 0.599424
    A112 = 0.010185 * X1 ** 3 - 0.161176 * X1 ** 2 + 0.904989 * X1 - 1.641389
    A113 = -0.015422 * X1 ** 3 + 0.220371 * X1 ** 2 - 1.084987 * X1 + 1.834167
    A121 = -0.0628667 * X1 ** 4 + 0.4989259 * X1 ** 3 + 0.52735 * X1 ** 2 - 10.7918672 * X1 + 16.616327
    A122 = 0.1140667 * X1 ** 4 - 0.8108963 * X1 ** 3 - 2.2186833 * X1 ** 2 + 25.1269741 * X1 - 37.7729778
    A123 = -0.0589333 * X1 ** 4 + 0.2639704 * X1 ** 3 + 3.1949667 * X1 ** 2 - 21.8126569 * X1 + 31.4113508
    A124 = 0.0107667 * X1 ** 4 + 0.0018704 * X1 ** 3 - 1.2494083 * X1 ** 2 + 6.9427931 * X1 - 10.2018992
    A131 = 0.192207 * X1 ** 3 - 2.787462 * X1 ** 2 + 12.507855 * X1 - 14.764856
    A132 = -0.350563 * X1 ** 3 + 5.222348 * X1 ** 2 - 23.974852 * X1 + 29.007851
    A133 = 0.237096 * X1 ** 3 - 3.535062 * X1 ** 2 + 16.368376 * X1 - 20.539908
    A134 = -0.067119 * X1 ** 3 + 0.966362 * X1 ** 2 - 4.407535 * X1 + 5.894703
    A11 = A111 * X2 ** 2 + A112 * X2 + A113
    A12 = A121 * X2 ** 3 + A122 * X2 ** 2 + A123 * X2 + A124
    A13 = A131 * X2 ** 3 + A132 * X2 ** 2 + A133 * X2 + A134
    AA111 = 17.945 * X1 ** 3 - 166.294 * X1 ** 2 + 489.799 * X1 - 493.142
    AA112 = -25.507 * X1 ** 3 + 236.275 * X1 ** 2 - 698.683 * X1 + 701.494
    AA113 = 9.077 * X1 ** 3 - 84.332 * X1 ** 2 + 249.983 * X1 - 250.787
    AA121 = -16.872 * X1 ** 3 + 156.399 * X1 ** 2 - 460.689 * X1 + 463.848
    AA122 = 24.015 * X1 ** 3 - 222.507 * X1 ** 2 + 658.027 * X1 - 660.665
    AA123 = -8.56 * X1 ** 3 + 79.549 * X1 ** 2 - 235.827 * X1 + 236.579
    AA11 = AA111 * X2 ** 2 + AA112 * X2 + AA113
    AA12 = AA121 * X2 ** 2 + AA122 * X2 + AA123
    AA1 = (AA11 * X3 + AA12) * (1 - X4) + 1.0
    A1 = (A11 * X4 ** 2 + A12 * X4 + A13) * AA1
    A2 = -1.402 * X4 ** 3 + 7.189 * X4 ** 2 - 10.993 * X4 + 9.45
    A31 = -7686.0287 * X2 ** 6 + 30131.5678 * X2 ** 5 - 49048.9664 * X2 ** 4 + 42480.7709 * X2 ** 3 - 20665.147 * X2 ** 2 + 5355.2035 * X2 - 577.8827
    A32 = 61639.9103 * X2 ** 6 - 241201.0598 * X2 ** 5 + 392579.5937 * X2 ** 4 - 340629.4699 * X2 ** 3 + 166348.6917 * X2 ** 2 - 43358.7938 * X2 + 4714.7918
    A33 = -130677.4903 * X2 ** 6 + 507996.2604 * X2 ** 5 - 826728.7127 * X2 ** 4 + 722677.104 * X2 ** 3 - 358360.7392 * X2 ** 2 + 95501.4948 * X2 - 10682.8619
    A34 = -110034.6584 * X2 ** 6 + 446051.22 * X2 ** 5 - 724186.4643 * X2 ** 4 + 599411.9264 * X2 ** 3 - 264294.7189 * X2 ** 2 + 58039.7328 * X2 - 4774.6414
    A35 = 709672.0656 * X2 ** 6 - 2803850.2395 * X2 ** 5 + 4553780.5017 * X2 ** 4 - 3888378.9905 * X2 ** 3 + 1839829.259 * X2 ** 2 - 457313.6939 * X2 + 46600.823
    A36 = -822735.9289 * X2 ** 6 + 3238899.7308 * X2 ** 5 - 5256636.5472 * X2 ** 4 + 4500543.147 * X2 ** 3 - 2143487.3508 * X2 ** 2 + 538548.1194 * X2 - 55751.1528
    A37 = 299122.8727 * X2 ** 6 - 1175773.1606 * X2 ** 5 + 1907356.1357 * X2 ** 4 - 1634256.8172 * X2 ** 3 + 780020.9393 * X2 ** 2 - 196679.7143 * X2 + 20467.0904
    AA311 = (
                        -17.102 * X2 ** 3 + 41.495 * X2 ** 2 - 33.234 * X2 + 8.8007) * X4 + 36.566 * X2 ** 3 - 89.203 * X2 ** 2 + 71.8 * X2 - 18.108
    AA31 = (-0.3767 * X1 ** 3 + 3.39 * X1 ** 2 - 10.356 * X1 + 11.588) * AA311
    AA32 = -0.0727 * X1 ** 2 + 0.7 * X1 - 1.2818
    XX4 = X4 - AA32
    AA3 = AA31 * (
                -1.05584 * XX4 ** 9 + 12.688 * XX4 ** 8 - 63.70534 * XX4 ** 7 + 172.84571 * XX4 ** 6 - 274.05701 * XX4 ** 5 + 257.68705 * XX4 ** 4 - 141.40915 * XX4 ** 3 + 44.13177 * XX4 ** 2 - 7.1654 * XX4 - 0.0495 * X1 ** 2 + 0.4518 * X1 - 0.61655)
    A3 = A31 * X4 ** 6 + A32 * X4 ** 5 + A33 * X4 ** 4 + A34 * X4 ** 3 + A35 * X4 ** 2 + A36 * X4 + A37 + AA3
    BWHAT = A1 / X5 * EXP(-A2 * (LOG(X5) - A3) ** 2 / 1.44)
    return BWHAT


def calculate_B_F(BD, BRTH, CB, DRAFT, KVC, LPP, OGD, OMEGA, PHI):
    """
    Skin friction damping B_F

    :param BD: Beam/DRAFT
    :param BRTH: Beam [m]
    :param CB: Block coefficient [-]
    :param DRAFT: DRAFT : ship draught [m]
    :param KVC = 1.14e-6  # Kinematic Viscosity Coefficient
    :param LPP: [m]
    :param OGD: OG/DRAFT
    :param OMEGA: Frequency of motion [rad/s]
    :param PHI: Roll angle [deg]
    :param CMID: Middship coefficient (A_m/(B*d) [-]

    :return: B_F
    """

    # There must be a typo in the original fortran code it was 102 instead of 1025!?
    RO = 1025  # Density of water

    TW = 2 * PI / OMEGA
    RF = DRAFT * ((0.887 + 0.145 * CB) * (1.7 + CB * BD) - 2.0 * OGD) / PI
    SF = LPP * (1.75 * DRAFT + CB * BRTH)
    CF = 1.328 * ((3.22 * RF ** 2 * (PHI * PI / 180) ** 2) / (TW * KVC)) ** -0.5
    BF = 4.0 / 3.0 / PI * RO * SF * RF ** 3 * (PHI * PI / 180) * OMEGA * CF
    BFHAT = BF / (RO * LPP * BRTH ** 3 * DRAFT * CB) * SQRT(BRTH / 2.0 / 9.81)
    return BFHAT


class SimplifiedIkedaInputError(ValueError): pass

limits_kawahara = {
    'CB'    : (0.5,0.85),
    r'B/d'  : (2.5,4.5),
    r'OG/d' : (-1.5,0.2),
    'CMID'    : (0.9,0.99),
    r'bBk/B': (0.01, 0.06),
    r'lBk/LPP': (0.05, 0.4),
    'OMEGA_hat': (0,1.0)
}  # Input limits for damping according to the original paper ny Kawahara

def _calculate_limit_compare_value(LPP, Beam, OG, lBK, bBK, OMEGA,
                                   DRAFT):
    inputs = {
        r'B/d': Beam / DRAFT,
        r'OG/d': OG / DRAFT,
        r'bBk/B': bBK / Beam,
        r'lBk/LPP': lBK / LPP,
        'OMEGA_hat': OMEGA * SQRT(Beam / 2 / 9.81)
    }
    return inputs

def verify_inputs(LPP,Beam,CB,CMID,OG,PHI,lBK,bBK,OMEGA,
                           DRAFT):
    inputs = {
        'LPP': LPP,
        'Beam': Beam,
        'CB': CB,
        'CMID': CMID,
        'OG': OG,
        'PHI': PHI,
        'BKL': lBK,
        'BKB': bBK,
        'OMEGA': OMEGA,
        'DRAFT': DRAFT,
    }
    limit_inputs = _calculate_limit_compare_value(LPP, Beam, OG, lBK, bBK, OMEGA,
                                                  DRAFT)

    inputs.update(limit_inputs)

    exclude_zero = 10*-10
    limits={
        'LPP'   : (exclude_zero,np.inf),
        'Beam'  : (exclude_zero,np.inf),
        'CB'    : (0.4,1.0),
        'CMID'  : (0.4,1.0),
        'OG'    : (-np.inf,np.inf),
        'PHI'   : (exclude_zero, np.inf),
        'BKL'   : (0,np.inf),
        'BKB'   : (0,np.inf),
        'OMEGA' : (exclude_zero,np.inf),
        'DRAFT' : (exclude_zero,np.inf)
    }
    limits.update(limits_kawahara)
    if lBK==0:
        # Remove bilge keel limit when the bilge keel is not present:
        if r'bBk/B' in limits:
            limits.pop(r'bBk/B')

        if r'lBk/LPP' in limits:
            limits.pop(r'lBk/LPP')

    for key,value in inputs.items():

        if not key in limits:
            continue

        lims = limits.get(key)

        if np.any(pd.isnull(value)):
            raise SimplifiedIkedaInputError('%s is NaN' % key)

        #if not np.all((lims[0] <= value) & (value <= lims[1])):
        #    raise SimplifiedIkedaInputError('%s has a bad value:%f is not in (%f,%f)' % (key,value, lims[0], lims[1]))

        if np.any((value - lims[0]) < -0.000001):
            raise SimplifiedIkedaInputError('%s:%f is too small  (<%f)' % (key, value, lims[0]))

        if np.any((lims[1] - value) < -0.000001):
            raise SimplifiedIkedaInputError('%s:%f is too large  (>%f)' % (key, value, lims[1]))

    if np.any((LPP/Beam > 100)):
        raise SimplifiedIkedaInputError('Lpp/Beam has bad ratio' % (LPP/Beam))

def _calculate_limit_value(LPP, Beam, DRAFT):

    beam_limit = np.array(limits_kawahara['B/d']) * DRAFT
    _beam=Beam
    if Beam < beam_limit[0]:
        _beam =beam_limit[0]
    elif Beam > beam_limit[1]:
        _beam = beam_limit[1]

    limits = {
        'CB': np.array(limits_kawahara['CB']),
        'CMID': np.array(limits_kawahara['CMID']),
        'Beam': np.array(limits_kawahara['B/d'])*DRAFT,
        'OG': np.array(limits_kawahara[r'OG/d'])*DRAFT,
        'BKB': np.array(limits_kawahara[r'bBk/B'])*_beam,
        'BKL': np.array(limits_kawahara[r'lBk/LPP'])*LPP,
        'OMEGA': np.array(limits_kawahara['OMEGA_hat'])/(SQRT(_beam / 2 / 9.81)),
    }

    return limits

def _limit_inputs(LPP,Beam,CB,CMID,OG,PHI,lBK,bBK,OMEGA,
                           DRAFT):
    inputs = {
        'LPP': LPP,
        'Beam': Beam,
        'CB': CB,
        'CMID': CMID,
        'OG': OG,
        'PHI': PHI,
        'BKL': lBK,
        'BKB': bBK,
        'OMEGA': OMEGA,
        'DRAFT': DRAFT,
    }

    limits = _calculate_limit_value(LPP=LPP, Beam=Beam, DRAFT=DRAFT)

    if lBK==0:
        # Remove bilge keel limit when the bilge keel is not present:
        if 'BKL' in limits:
            limits.pop('BKL')

        if 'BKB' in limits:
            limits.pop('BKB')

    outputs = {}
    for key, value in inputs.items():

        if not key in limits:
            outputs[key] = inputs[key]
            continue

        limit = limits[key]
        if inputs[key] < limit[0]:
            outputs[key] = limit[0]
        elif inputs[key] > limit[1]:
            outputs[key] = limit[1]
        else:
            outputs[key] = inputs[key]

    return outputs

