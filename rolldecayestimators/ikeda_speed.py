"""
This is a translation of Carl-Johans implementation in Matlab to Python
"""
import os.path
import numpy as np
from numpy import tanh, exp, sqrt, pi, sin, cos, arctan
import pandas as pd
import scipy.interpolate

dir_path = os.path.dirname(__file__)
base_path = os.path.split(dir_path)[0]
data_path = os.path.join(base_path,'rolldecayestimators','Bw0_S175.csv')
data = pd.read_csv(data_path, sep=';')
f_interpolation = scipy.interpolate.interp1d(data['w_vec'],data['b44_vec'], kind='cubic')

def Bw0_S175(w):
    Bw0 = f_interpolation(w)
    return Bw0

def Bw_S175(w, V, d, g=9.81):

    Bw0 = Bw0_S175(w)
    OMEGA = w * V / g
    zeta_d = w**2 * d / g
    A1 = 1 + zeta_d**(-1.2) * exp(-2 * zeta_d)
    A2 = 0.5 + zeta_d**(-1) * exp(-2 * zeta_d)

    Bw_div_Bw0 = 0.5 * (
                ((A1 + 1) + (A2 - 1) * tanh(20 * (OMEGA - 0.3))) + (2 * A1 - A2 - 1) * exp(-150 * (OMEGA - 0.25)**2))
    bw44 = Bw0 * Bw_div_Bw0

    return bw44


def bilge_keel(wE, fi_a, V, B, d, A, bBK, R, g, OG, Ho, ra):
    """
    ITTC
    definitions

    B44BK = B44BK_N0 + B44BK_H0 + B44BK_L + B44BK_W
    """

    tata = A / (B * d);


    # Normal Force component(ITTC)
    f = 1 + 0.3 * exp(-160 * (1 - tata));
    l = d * sqrt((Ho - (1 - sqrt(2) / 2) * R / d) ** 2 + (1 - OG / d - (1 - sqrt(2) / 2) * R / d) ** 2) # distance from CoG to tip of bilge keel
    CD = 22.5 * bBK / (pi * l * fi_a * f) + 2.4;
    Bp44BK_N0 = 8 / (3 * pi) * ra * l ** 3 * wE * fi_a * bBK * f * CD;

    # #
    # Hull pressure component
    So = 0.3 * pi * l * fi_a * f / bBK + 1.95;
    m1 = R / d;
    m2 = OG / d;
    m3 = 1 - m1 - m2;
    m4 = Ho - m1;
    m5 = (0.414 * Ho + 0.0651 * m1 ** 2 - (0.382 * Ho + 0.0106) * m1) / ((Ho - 0.215 * m1) * (1 - 0.215 * m1));
    m6 = (0.414 * Ho + 0.0651 * m1 ** 2 - (0.382 + 0.0106 * Ho) * m1) / ((Ho - 0.215 * m1) * (1 - 0.215 * m1));

    if So > 0.25 * pi * R:
        m7 = So / d - 0.25 * pi * m1;
    else:
        m7 = 0;

    if So > 0.25 * pi * R:
        m8 = m7 + 0.414 * m1;
    else:
        m8 = m7 + 1.414 * m1 * (1 - cos(So / R));

    Ao = (m3 + m4) * m8 - m7 ** 3;
    Bo = m2 ** 2 / (3 * (Ho - 0.215 / m1)) + (1 - m1) ** 2 * (2 * m3 - m2) / (6 * (1 - 0.215 * m1)) + m1 * (
                m3 * m5 + m4 * m6);
    Cp_minus = -22.5 * bBK / (pi * l * fi_a * f) - 1.2;
    Cp_plus = 1.2;

    Bp44BK_H0 = 4 / (3 * pi) * ra * l ** 2 * wE * fi_a * d ** 2 * (-Ao * Cp_minus + Bo * Cp_plus);

    # # Ikeda 1994 bilge keel generated Lift # Fartberoende??
    l1 = l + bBK / 2; # Lift Force
    u = l1 * fi_a * wE; # tangential velocity
    alpha = arctan(u / V); # flow velocity?
    Vr = sqrt(V ** 2 + u ** 2); # -'' -

    LBK = pi * ra * alpha * Vr ** 2 * bBK ** 2 / 2; # lift

    B44BK_L = 2 * LBK * l1 / (fi_a * wE); #

    # #
    # wave making contribution from bilge keels, normally very small...,
    # non - dimensional!!!!!

    C_BK = bBK / B; # estimation according to ITTC
    lBK = l - bBK; # needs to be verified!!!!
    fi = fi_a;

    dBK = lBK * ((2 * d / B) / sqrt(1 + (2 * d / B) ** 2) * cos(fi) - sin(fi) / (1 + (2 * d / B) ** 2));

    B44BKW0 = C_BK * exp(-wE ** 2 / g * dBK); # non dimensional  wave damping from BK, ITTC
    return Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0

def  frictional(wE,fi_a,V,B, d, OG, ra, Cb, L, visc =   1.15*10**-6):
    # ITTC

    Sf   = L*(1.7*d+Cb*B); # Wetted surface approx
    r_f  = 1/pi*((0.887+0.145*Cb)*(Sf/L)+2*OG);

    Rn = 0.512*(r_f/fi_a)**2*wE/visc;
    Cf = 1.328*Rn**-0.5+0.14*Rn**-0.114;
    B44F0=0.5*ra*r_f**3*Sf*Cf;
    B44F=B44F0*8/(3*pi)*fi_a*wE*(1+4.1*V/(wE*L))
    return B44F

def hull_lift(V,B, d, OG, ra, L):

    lo   = 0.3*d;
    lR   = 0.5*d;
    K    = 0.1;         #!! depends on CM

    kN   = 2*pi*d/L+K*(4.1*B/L-0.045);
    B44L = ra/2*V*L*d*kN*lo*lR*(1+1.4*OG/lR+0.7*OG**2/(lo*lR))

    return B44L