import pytest
from rolldecayestimators import ikeda_speed_S175 as ikeda  # Note that this one is slightly different (I'm not 100% sure wish one is correct)
from numpy import sqrt, pi
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal,assert_allclose
import matplotlib.pyplot as plt

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
    Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0 = ikeda.bilge_keel(w=wE, fi_a=fi_a, V=V, B=B, d=d, A=A, bBK=bBK, R=R, g=g, OG=OG, Ho=Ho, ra=ra)

    assert_allclose(Bp44BK_N0, 1.384980250651594e+05, rtol=0.001)
    assert_allclose(Bp44BK_H0, 6.137595375551779e+05, rtol=0.001)
    assert_allclose(B44BK_L, 7.221062603618902e+04, rtol=0.001)
    assert_allclose(B44BKW0, 0.014575557482580, rtol=0.001)

def test_friction():
    V = 0  ## Ship speed
    T = 20
    wE = 2 * np.pi * 1 / T  # circular frequency
    fi_a = 5 * pi / 180;  # roll amplitude !!rad??
    B44F = ikeda.frictional(w=wE, fi_a=fi_a, V=V, B=B, d=d, OG=OG, ra=ra, Cb=Cb, L=L, visc=visc)
    assert_allclose(B44F, 7.015809420450178e+05, rtol=0.001)

def test_hull_lift():
    V = 0  ## Ship speed
    B44L = ikeda.hull_lift(V=V, B=B, d=d, OG=OG, ra=ra, L=L, A=A)
    assert_allclose(B44L, 0, rtol=0.001)

def test_calculate_B44():
    V = 0  ## Ship speed
    T = 20
    wE = 2 * np.pi * 1 / T  # circular frequency
    fi_a = 5 * pi / 180;  # roll amplitude !!rad??
    Bw0 = ikeda.Bw0_S175(w=wE)
    B44,BW44,B44_BK,B44F,B44L = ikeda.calculate_B44(w=wE, V=V, d=d, Bw0=Bw0, fi_a=fi_a,  B=B,  A=A, bBK=bBK, R=R, OG=OG, Ho=Ho, ra=ra, Cb=Cb, L=L, LBK=LBK, visc = visc, g=g)

    assert_allclose(BW44, 5.6885e+06, rtol=0.001)
    assert_allclose(B44_BK, 3.298347899067595e+07, rtol=0.001)
    assert_allclose(B44F, 7.015809420450178e+05, rtol=0.001)
    assert_allclose(B44L, 0, rtol=0.001)

def test_calculate_B44_speeds():

    V = np.arange(1,17+1)

    T = 20
    wE = 2 * np.pi * 1 / T  # circular frequency
    fi_a = 5 * pi / 180;  # roll amplitude !!rad??
    Bw0 = ikeda.Bw0_S175(w=wE)

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

    results = inputs.apply(func=ikeda.calculate_B44_series, axis=1)
    # Wave:
    BW44_matlab = [5699056.52723334, 5768890.58302138, 6103827.43105922, 7254068.96665544, 10031079.9833426, 14587981.3683833,
     19308156.9962379, 21828526.5996131, 21922623.0633245, 20923134.3478225, 19428867.5785877, 18312620.2325888,
     17861271.9724633, 17755006.1702713, 17743513.2039722, 17745347.1366789, 17746686.6549155]
    assert_allclose(results['B_W'], BW44_matlab, rtol=0.001)

    # B44_BK
    B44_BK_matlab = [33036485.1748026,33143743.2149632,33254853.2019460,33366974.1270120,33479504.5019107,33592240.6642520,33705094.7338643,33818022.6109119,33930999.7420341,34044011.3743272,34157048.1104209,34270103.6810097,34383173.7435435,34496255.1949657,34609345.7589930,34722443.7279897,34835547.7958544]
    assert_allclose(results['B_BK'], B44_BK_matlab, rtol=0.001)

    # B44F:
    B44F_matlab = [753901.662857896,806222.383670775,858543.104483653,910863.825296531,963184.546109410,1015505.26692229,1067825.98773517,1120146.70854805,1172467.42936092,1224788.15017380,1277108.87098668,1329429.59179956,1381750.31261244,1434071.03342532,1486391.75423819,1538712.47505107,1591033.19586395]
    assert_allclose(results['B_F'], B44F_matlab, rtol=0.001)

    # Lift:
    B44L_matlab = [4595742.29457220,9191484.58914441,13787226.8837166,18382969.1782888,22978711.4728610,27574453.7674332,32170196.0620054,36765938.3565776,41361680.6511498,45957422.9457220,50553165.2402943,55148907.5348664,59744649.8294386,64340392.1240109,68936134.4185831,73531876.7131553,78127619.0077275]
    assert_allclose(results['B_L'], B44L_matlab, rtol=0.001)

    fig,ax=plt.subplots()
    result_nondim = results*ND_factor
    result_nondim.plot.area(y = ['B_BK','B_L','B_W','B_F'], ax=ax)
    plt.show()
