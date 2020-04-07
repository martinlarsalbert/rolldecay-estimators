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

def calculate_roll_damping(LPP,Beam,CB,CMID,OG,PHI,lBK,bBK,OMEGA,
                           DRAFT):
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
    :param lBK: length of bilge keel [m]
    :param bBK: height of bilge keel [m]
    :param OMEGA: Frequency of motion [rad/s]
    :param DRAFT: DRAFT : ship draught [m]
    :param OMEGAHAT:
    :return: B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT
     Nondimensional damping:
    B44HAT: Total
    BFHAT: Friction
    BWHAT: Wave
    BEHAT: Eddy
    BBKHAT: Bilge keel
    """

    LBKL=lBK/LPP
    BD = Beam/DRAFT
    OGD = OG/DRAFT
    BBKB = bBK/Beam
    BRTH = Beam
    OMEGA/= 2  # Magic factor!??? This value seemt o give better results..?
    OMEGAHAT = OMEGA * SQRT(BRTH / 2 / 9.81)
    TW = 2 * PI / OMEGA

    #There must be a typo in the original fortran code it was 102 instead of 1025!?
    RO=1025  # Density of water
    KVC = 1.14e-6  # Kinematic Viscosity Coefficient

    #*** Frictional Component ***
    RF=DRAFT*((0.887+0.145*CB)*(1.7+CB*BD)-2.0*OGD)/PI
    SF=LPP*(1.75*DRAFT+CB*BRTH)
    CF=1.328*((3.22*RF**2*(PHI*PI/180)**2)/(TW*KVC))**-0.5
    BF=4.0/3.0/PI*RO*SF*RF**3*(PHI*PI/180)*OMEGA*CF
    BFHAT=BF/(RO*LPP*BRTH**3*DRAFT*CB)*SQRT(BRTH/2.0/9.81)

    #*** Wave Component ***
    X1=BD ; X2=CB ; X3=CMID
    X5=OMEGAHAT
    X4=1-OGD
    A111=-0.002222*X1**3+0.040871*X1**2-0.286866*X1+0.599424
    A112=0.010185*X1**3-0.161176*X1**2+0.904989*X1-1.641389
    A113=-0.015422*X1**3+0.220371*X1**2-1.084987*X1+1.834167
    A121=-0.0628667*X1**4+0.4989259*X1**3+0.52735*X1**2-10.7918672*X1+16.616327
    A122=0.1140667*X1**4-0.8108963*X1**3-2.2186833*X1**2+25.1269741*X1-37.7729778
    A123=-0.0589333*X1**4+0.2639704*X1**3+3.1949667*X1**2-21.8126569*X1+31.4113508
    A124=0.0107667*X1**4+0.0018704*X1**3-1.2494083*X1**2+6.9427931*X1-10.2018992
    A131=0.192207*X1**3-2.787462*X1**2+12.507855*X1-14.764856
    A132=-0.350563*X1**3+5.222348*X1**2-23.974852*X1+29.007851
    A133=0.237096*X1**3-3.535062*X1**2+16.368376*X1-20.539908
    A134=-0.067119*X1**3+0.966362*X1**2-4.407535*X1+5.894703

    A11=A111*X2**2+A112*X2+A113
    A12=A121*X2**3+A122*X2**2+A123*X2+A124
    A13=A131*X2**3+A132*X2**2+A133*X2+A134

    AA111=17.945*X1**3-166.294*X1**2+489.799*X1-493.142
    AA112=-25.507*X1**3+236.275*X1**2-698.683*X1+701.494
    AA113=9.077*X1**3-84.332*X1**2+249.983*X1-250.787
    AA121=-16.872*X1**3+156.399*X1**2-460.689*X1+463.848
    AA122=24.015*X1**3-222.507*X1**2+658.027*X1-660.665
    AA123=-8.56*X1**3+79.549*X1**2-235.827*X1+236.579

    AA11=AA111*X2**2+AA112*X2+AA113
    AA12=AA121*X2**2+AA122*X2+AA123

    AA1=(AA11*X3+AA12)*(1-X4)+1.0

    A1=(A11*X4**2+A12*X4+A13)*AA1
    A2=-1.402*X4**3+7.189*X4**2-10.993*X4+9.45

    A31=-7686.0287*X2**6+30131.5678*X2**5-49048.9664*X2**4+42480.7709*X2**3-20665.147*X2**2+5355.2035*X2-577.8827
    A32=61639.9103*X2**6-241201.0598*X2**5+392579.5937*X2**4-340629.4699*X2**3+166348.6917*X2**2-43358.7938*X2+4714.7918
    A33=-130677.4903*X2**6+507996.2604*X2**5-826728.7127*X2**4+722677.104*X2**3-358360.7392*X2**2+95501.4948*X2-10682.8619
    A34=-110034.6584*X2**6+446051.22*X2**5-724186.4643*X2**4+599411.9264*X2**3-264294.7189*X2**2+58039.7328*X2-4774.6414
    A35=709672.0656*X2**6-2803850.2395*X2**5+4553780.5017*X2**4-3888378.9905*X2**3+1839829.259*X2**2-457313.6939*X2+46600.823
    A36=-822735.9289*X2**6+3238899.7308*X2**5-5256636.5472*X2**4+4500543.147*X2**3-2143487.3508*X2**2+538548.1194*X2-55751.1528
    A37=299122.8727*X2**6-1175773.1606*X2**5+1907356.1357*X2**4-1634256.8172*X2**3+780020.9393*X2**2-196679.7143*X2+20467.0904

    AA311=(-17.102*X2**3+41.495*X2**2-33.234*X2+8.8007)*X4+36.566*X2**3-89.203*X2**2+71.8*X2-18.108

    AA31=(-0.3767*X1**3+3.39*X1**2-10.356*X1+11.588)*AA311
    AA32=-0.0727*X1**2+0.7*X1-1.2818

    XX4=X4-AA32

    AA3=AA31*(-1.05584*XX4**9+12.688*XX4**8-63.70534*XX4**7+172.84571*XX4**6-274.05701*XX4**5+257.68705*XX4**4-141.40915*XX4**3+44.13177*XX4**2-7.1654*XX4-0.0495*X1**2+0.4518*X1-0.61655)

    A3=A31*X4**6+A32*X4**5+A33*X4**4+A34*X4**3+A35*X4**2+A36*X4+A37+AA3

    BWHAT=A1/X5*EXP(-A2*(LOG(X5)-A3)**2/1.44)

    #*** Eddy Component ***
    FE1=(-0.0182*CB+0.0155)*(BD-1.8)**3
    FE2=-79.414*CB**4+215.695*CB**3-215.883*CB**2+93.894*CB-14.848
    AE=FE1+FE2
    BE1=(3.98*CB-5.1525)*(-0.2*BD+1.6)*OGD*((0.9717*CB**2-1.55*CB+0.723)*OGD+0.04567*CB+0.9408)
    BE2=(0.25*OGD+0.95)*OGD-219.2*CB**3+443.7*CB**2-283.3*CB+59.6
    BE3=-15*CB*BD+46.5*CB+11.2*BD-28.6
    CR=AE*EXP(BE1+BE2*CMID**BE3)
    BEHAT=4.0*OMEGAHAT*PHI*PI/180/(3.0*PI*CB*BD**3.0)*CR

    #*** Bilge Keel Component ***
    if (lBK==0):
        BBKHAT=0.0
    else:
        FBK1=(-0.3651*CB+0.3907)*(BD-2.83)**2-2.21*CB+2.632
        FBK2=0.00255*PHI**2+0.122*PHI+0.4794
        FBK3=(-0.8913*BBKB**2-0.0733*BBKB)*LBKL**2+(5.2857*BBKB**2-0.01185*BBKB+0.00189)*LBKL
        ABK=FBK1*FBK2*FBK3
        BBK1=(5.0*BBKB+0.3*BD-0.2*LBKL+0.00125*PHI**2-0.0425*PHI-1.86)*OGD
        BBK2=-15.0*BBKB+1.2*CB-0.1*BD-0.0657*OGD**2+0.0586*OGD+1.6164
        BBK3=2.5*OGD+15.75
        BBKHAT=ABK*EXP(BBK1+BBK2*CMID**BBK3)*OMEGAHAT

    #*** Total Roll Damping ***
    B44HAT=BFHAT+BWHAT+BEHAT+BBKHAT

    return B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT