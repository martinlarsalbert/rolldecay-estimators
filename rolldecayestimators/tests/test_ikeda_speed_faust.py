from rolldecayestimators import ikeda_speed as ikeda
from numpy import sqrt, pi
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import pytest

ScaleF =  1 #%/29.565;                # Scale Factor [-]
visc =   1.15*10**-6;                 # [m2/s], kinematic viscosity
Cb   =   0.61;                        # Block coeff
L    =   220*ScaleF;                  # Length
vcg  =   14.4*ScaleF;                 # roll axis (vertical centre of gravity) [m]
vcg  =   14.9*ScaleF;                 # roll axis (vertical centre of gravity) [m]
B    =   32.26*ScaleF;                # Breadth of hull [m]
d    =   9.5*ScaleF;                  # Draught of hull [m]
A    =   0.93*B*d;                    # Area of cross section of hull [m2]
bBK  =   0.4*ScaleF;                  # breadth of Bilge keel [m] !!(=height???)
R    =   5*ScaleF;                    # Bilge Radis
g    =   9.81;
C_mid=   0.93;

OG = -1*(vcg-d)#*0.8;                    # distance from roll axis to still water level
Ho = B/(2*d);                  # half breadth to draft ratio
ra   = 1025;                   # density of water

#locals
LBK  = L/4;                    # Approx
disp = L*B*d*Cb;               # Displacement

# variables!!
T=27.6*sqrt(ScaleF); wE   = 2*pi*1/T;        # circular frequency
fi_a =   10*pi/180;            # roll amplitude !!rad??
V    =   0;                  # Speed

ND_factorB = sqrt(B/(2*g))/(ra*disp*(B**2));   # Nondimensiolizing factor of B44
ND_factor = wE/(2*ra*disp*g*(16*ScaleF-vcg));   # Nondimensiolizing factor of B44

def test_Bw0():
    Bw0=ikeda.Bw0_faust(w=0.227651641564478)
    assert_allclose(Bw0,1.763939936010376e+06, rtol=0.001)

def test_bw44_V0():
    V = 0  ## Ship speed
    T = 27.6 * sqrt(ScaleF);
    wE = 2 * pi * 1 / T;  # circular frequency


    bw44 = ikeda.Bw_faust(w=wE, V=V, d=d)

    assert_allclose(bw44, 1.523005619852937e+07, rtol=0.001)

def test_bilge_keel():
    V = 0  ## Ship speed
    T = 27.6 * sqrt(ScaleF);
    wE = 2 * pi * 1 / T;  # circular frequency
    fi_a = 10 * pi / 180;  # roll amplitude !!rad??
    Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0 = ikeda.bilge_keel(w=wE, fi_a=fi_a, V=V, B=B, d=d, A=A, bBK=bBK, R=R, g=g, OG=OG, Ho=Ho, ra=ra)

    assert_allclose(Bp44BK_N0, 3.509033728796537e+05, rtol=0.001)
    assert_allclose(Bp44BK_H0, 1.057765842625329e+06, rtol=0.001)
    assert_allclose(B44BK_L, 2.607161328890767e+05, rtol=0.001)
    assert_allclose(B44BKW0, 0.011960200569156, rtol=0.001)

@pytest.mark.skip('This one does not work yet')
def test_friction():
    V = 0  ## Ship speed
    T = 27.6 * sqrt(ScaleF);
    wE = 2 * pi * 1 / T;  # circular frequency
    fi_a = 10 * pi / 180;  # roll amplitude !!rad??
    B44F = ikeda.frictional(w=wE, fi_a=fi_a, V=V, B=B, d=d, OG=OG, ra=ra, Cb=Cb, L=L, visc=visc)
    assert_allclose(B44F, 5.794039754129194e+06, rtol=0.001)

def test_hull_lift():
    V = 10  ## Ship speed
    B44L = ikeda.hull_lift(V=V, B=B, d=d, OG=OG, ra=ra, L=L, A=A)
    assert_allclose(B44L, 1.734463413980598e+08, rtol=0.001)

@pytest.mark.skip('This one does not work yet')
def test_calculate_B44():
    V = 10  ## Ship speed
    T = 27.6 * sqrt(ScaleF);
    wE = 2 * pi * 1 / T;  # circular frequency
    fi_a = 10 * pi / 180;  # roll amplitude !!rad??
    Bw0 = ikeda.Bw0_faust(w=wE)
    B44,BW44,B44_BK,B44F,B44L = ikeda.calculate_B44(w=wE, V=V, d=d, Bw0=Bw0, fi_a=fi_a,  B=B,  A=A, bBK=bBK, R=R, OG=OG,
                                                    Ho=Ho, ra=ra, Cb=Cb, L=L, LBK=LBK, visc = visc, g=g, Bw_div_Bw0_max=np.inf)

    assert_allclose(BW44, 44060267.568187, rtol=0.001)
    assert_allclose(B44_BK, 79564994.97049402, rtol=0.001)
    assert_allclose(B44F, 10537244.373232083, rtol=0.001)
    assert_allclose(B44L, 173446341.3980598, rtol=0.001)

@pytest.mark.skip('This one does not work yet')
def test_calculate_B44_speeds():

    V = np.arange(0,10+1)

    T = 27.6 * sqrt(ScaleF);
    wE = 2 * pi * 1 / T;  # circular frequency
    fi_a = 10 * pi / 180;  # roll amplitude !!rad??
    Bw0 = ikeda.Bw0_faust(w=wE)

    inputs = pd.DataFrame(index=V)
    inputs['V']=V
    inputs['w']=wE
    inputs['fi_a']=fi_a
    inputs['Bw0']=Bw0
    inputs['d']=d
    inputs['B']=B
    inputs['A']=A
    inputs['BKB']=bBK
    inputs['R_b']=R
    inputs['OG']=OG
    inputs['Ho']=Ho
    inputs['rho']=ra
    inputs['Cb']=Cb
    inputs['L']=L
    inputs['LBK']=LBK
    inputs['visc']=visc
    inputs['g']=g

    results = inputs.apply(func=ikeda.calculate_B44_series, Bw_div_Bw0_max=np.inf, axis=1)

    # This one is no longer valid
    # Wave:
    #BW44_matlab = [15230056.1985294,15245650.7598695,15312025.5743705,15551094.8502096,16275924.5707451,18116404.2772792,22004594.1044359,28771214.8371744,38312799.5815772,48887362.9192075,57522599.4248384]
    #assert_allclose(results['B_W'], BW44_matlab, rtol=0.001)


    # B44_BK
    B44_BK_matlab = [77737522.9856631,77764104.9949679,77935523.2118694,78129653.2817479,78330461.4926025,78534065.8999322,78739095.8600784,78944948.7197441,79151318.9709843,79358035.4695565,79564994.9704940]
    assert_allclose(results['B_BK'], B44_BK_matlab, rtol=0.001)

    # B44F:
    B44F_matlab = [5794039.75412919,6268360.21603948,6742680.67794977,7217001.13986006,7691321.60177035,8165642.06368064,8639962.52559093,9114282.98750122,9588603.44941150,10062923.9113218,10537244.3732321]
    assert_allclose(results['B_F'], B44F_matlab, rtol=0.001)

    # Lift:
    B44L_matlab = [0,17344634.1398060,34689268.2796119,52033902.4194179,69378536.5592239,86723170.6990299,104067804.838836,121412438.978642,138757073.118448,156101707.258254,173446341.398060]
    assert_allclose(results['B_L'], B44L_matlab, rtol=0.001)

    fig,ax=plt.subplots()
    result_nondim = results*ND_factor
    result_nondim.plot.area(y = ['B_BK','B_L','B_W','B_F'], ax=ax)
    plt.show()

