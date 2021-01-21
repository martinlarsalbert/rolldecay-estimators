"""
This is a translation of Carl-Johans implementation in Matlab to Python
"""
import os.path
import numpy as np
from numpy import tanh, exp, sqrt, pi, sin, cos, arccos, min, max
import pandas as pd
import scipy.interpolate
from copy import copy
import matplotlib.pyplot as plt

dir_path = os.path.dirname(__file__)
base_path = os.path.split(dir_path)[0]

data_path_S175 = os.path.join(base_path, 'rolldecayestimators', 'Bw0_S175.csv')
data_S175 = pd.read_csv(data_path_S175, sep=';')
f_interpolation_S175 = scipy.interpolate.interp1d(data_S175['w_vec'], data_S175['b44_vec'], kind='cubic')

data_path_faust = os.path.join(base_path, 'rolldecayestimators', 'Bw0_faust.csv')
data_faust = pd.read_csv(data_path_faust, sep=';')
f_interpolation_faust = scipy.interpolate.interp1d(data_faust['w_vec'], data_faust['b44_vec'], kind='cubic')
from scipy.integrate import simps

def Bw0_S175(w):
    Bw0 = f_interpolation_S175(w)
    return Bw0

def Bw0_faust(w):
    Bw0 = f_interpolation_faust(w)
    return Bw0

def Bw_S175(w, V, d, g=9.81):
    Bw0 = Bw0_S175(w)
    BW44 = Bw(w, V, d, Bw0, g=9.81)

    return BW44

def Bw_faust(w, V, d, g=9.81):
    Bw0 = Bw0_faust(w)
    BW44 = Bw(w, V, d, Bw0, g=9.81, speed_correction=B_W_speed_correction_factor)

    return BW44

def B_W_speed_correction_factor(w, V, d, g=9.81):
    """
        Wave damping speed correction
        Parameters
        ----------
        w
            "omega" frequency of motion [rad/s]
        V
            ship speed [m/s]
        d
            ship draught [m]
        g
            gravity

        Returns
        -------
        Bw_div_Bw0
            Bw_div_Bw0 = B_W/B_W0

        """
    tau=w*V/g
    zeta_d=w**2*d/g
    A1=1+zeta_d**(-1.2)*exp(-2*zeta_d);
    A2=0.5+zeta_d**(-1)*exp(-2*zeta_d);

    b = 20*0.3  # Unsure about this one...
    Bw_div_Bw0=0.5*(((A1+1)+(A2-1)*tanh(20*tau-b))
                    +(2*A1-A2-1)*exp(-150*(tau-0.25)**2))

    return Bw_div_Bw0

def B_W_speed_correction_factor_ikeda(w, V, d, g=9.81):
    """
        Wave damping speed correction
        Parameters
        ----------
        w
            "omega" frequency of motion [rad/s]
        V
            ship speed [m/s]
        d
            ship draught [m]
        g
            gravity

        Returns
        -------
        Bw_div_Bw0
            Bw_div_Bw0 = B_W/B_W0

        """
    tau=w*V/g
    zeta_d=w**2*d/g
    A1=1+zeta_d**(-1.2)*exp(-2*zeta_d);
    A2=0.5+zeta_d**(-1)*exp(-2*zeta_d);

    b=20*0.3
    Bw_div_Bw0=0.5*((A2+1)+(A2-1)*tanh(20*tau-b)
                    +(2*A1-A2-1)*exp(-150*(tau-0.25)**2))

    return Bw_div_Bw0

def B_W_speed_correction_factor_journee(w, V, d, g=9.81):
    """
        Wave damping speed correction
        Parameters
        ----------
        w
            "omega" frequency of motion [rad/s]
        V
            ship speed [m/s]
        d
            ship draught [m]
        g
            gravity

        Returns
        -------
        Bw_div_Bw0
            Bw_div_Bw0 = B_W/B_W0

        """
    tau=w*V/g
    zeta_d=w**2*d/g
    A1=1+zeta_d**(-1.2)*exp(-2*zeta_d);
    A2=0.5+zeta_d**(-1)*exp(-2*zeta_d);

    Bw_div_Bw0 = 0.5 * (A2 + 1 + (A2 - 1) * tanh(20 * (tau - 0.3))
                        + (2 * A1 - A2 - 1) * exp(-150 * (w - 0.25) ** 2)) - 1

    return Bw_div_Bw0


def Bw(w, V, d, Bw0, g=9.81, Bw_div_Bw0_max=12, speed_correction=B_W_speed_correction_factor_ikeda):
    """
    Wave damping speed correction
    Parameters
    ----------
    w
        "omega" frequency of motion [rad/s]
    V
        ship speed [m/s]
    d
        ship draught [m]
    Bw0
        wave roll damping at zero speed
    g
        gravity

    Returns
    -------
    BW44
        wave roll camping at speed

    """

    Bw_div_Bw0 = speed_correction(w=w, V=V, d=d, g=g)

    # Proposing a limit here:
    if Bw_div_Bw0>Bw_div_Bw0_max:
        Bw_div_Bw0=Bw_div_Bw0_max

    BW44 = Bw0 * Bw_div_Bw0
    return BW44


def bilge_keel(w, fi_a, V, B, d, A, bBK, R, g, OG, Ho, ra):
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
    Bp44BK_N0 = 8 / (3 * pi) * ra * l ** 3 * w * fi_a * bBK * f * CD;

    # #
    # Hull pressure component
    #So = 0.3 * pi * l * fi_a * f / BKB + 1.95; # S175 implementation
    So = (0.3 * pi * l * fi_a * f / bBK + 1.95) * bBK;
    m1 = R / d;
    m2 = OG / d;
    m3 = 1 - m1 - m2;
    m4 = Ho - m1;
    m5 = (0.414 * Ho + 0.0651 * m1 ** 2 - (0.382 * Ho + 0.0106) * m1) / ((Ho - 0.215 * m1) * (1 - 0.215 * m1));
    m6 = (0.414 * Ho + 0.0651 * m1 ** 2 - (0.382 + 0.0106 * Ho) * m1) / ((Ho - 0.215 * m1) * (1 - 0.215 * m1));


    m7 = np.where(So > 0.25 * pi * R,
                  So / d - 0.25 * pi * m1,
                  0)

    m8 = np.where(So > 0.25 * pi * R,
                  m7 + 0.414 * m1,
                  m7 + 1.414 * m1 * (1 - cos(So / R)))

    Ao = (m3 + m4) * m8 - m7 ** 3;
    Bo = m2 ** 2 / (3 * (Ho - 0.215 / m1)) + (1 - m1) ** 2 * (2 * m3 - m2) / (6 * (1 - 0.215 * m1)) + m1 * (
                m3 * m5 + m4 * m6);
    Cp_minus = -22.5 * bBK / (pi * l * fi_a * f) - 1.2;
    Cp_plus = 1.2;

    Bp44BK_H0 = 4 / (3 * pi) * ra * l ** 2 * w * fi_a * d ** 2 * (-Ao * Cp_minus + Bo * Cp_plus);

    # # Ikeda 1994 bilge keel generated Lift # Fartberoende??
    l1 = l + bBK / 2; # Lift Force
    u = l1 * fi_a * w; # tangential velocity
    alpha = np.arctan2(u,V); # flow velocity?
    Vr = sqrt(V ** 2 + u ** 2); # -'' -

    LBK = pi * ra * alpha * Vr ** 2 * bBK ** 2 / 2; # lift

    B44BK_L = 2 * LBK * l1 / (fi_a * w); #

    # #
    # wave making contribution from bilge keels, normally very small...,
    # non - dimensional!!!!!

    C_BK = bBK / B; # estimation according to ITTC
    lBK = l - bBK; # needs to be verified!!!!
    fi = fi_a;

    #dBK = BKL * ((2 * d / B) / sqrt(1 + (2 * d / B) ** 2) * cos(fi) - sin(fi) / (1 + (2 * d / B) ** 2));  # S175 implementation
    dBK = lBK * ((2 * d / B) / sqrt(1 + (2 * d / B)**2) * cos(fi) - sin(fi) / sqrt((1 + (2 * d / B)**2)));

    B44BKW0 = C_BK * exp(-w ** 2 / g * dBK); # non dimensional  wave damping from BK, ITTC
    return Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0

def frictional_carl_johan(w, fi_a, V, B, d, OG, ra, Cb, L, visc =1.15 * 10 ** -6):
    # ITTC

    Sf   = L*(1.7*d+Cb*B); # Wetted surface approx
    #r_f  = 1/pi*((0.887+0.145*Cb)*(Sf/L)+2*OG);  # S175
    r_f = 1 / pi * ((0.887 + 0.145 * Cb) * (Sf / L) - 2 * OG);

    Rn = 0.512 * (r_f/fi_a) ** 2 * w / visc;
    Cf = 1.328*Rn**-0.5+0.14*Rn**-0.114;
    B44F0=0.5*ra*r_f**3*Sf*Cf;
    B44F= B44F0 * 8 / (3*pi) * fi_a * w * (1 + 4.1 * V / (w * L))
    return B44F

def frictional(w, fi_a, V, B, d, OG, ra, Cb, L, visc =1.15 * 10 ** -6, Sf=None):
    """
    Frictional damping according to:
    Falzarano, Jeffrey, Somayajula, Abhilash, Seah, Robert,
    2015. An overview of the prediction methods for roll damping of ships.
    Ocean Systems Engineering 5, 55–76. https://doi.org/10.12989/OSE.2015.5.2.055


    Parameters
    ----------
    w
    fi_a
    V
    B
    d
    OG
    ra
    Cb
    L
    visc

    Returns
    -------
    B44F

    """

    if Sf is None:
        Sf   = L*(1.7*d+Cb*B); # Wetted surface approx

    r_f = 1 / pi * ((0.887 + 0.145 * Cb) * (Sf / L) - 2 * OG)  # (Kato, 1958)

    Rn = 0.512 * r_f**2 * fi_a**2 * w / visc

    Cf = 1.328*sqrt(2*pi*visc/(3.22*r_f**2*fi_a**2*w))

    #B44F0 = 4/(3*pi)*ra*Sf*r_f**3*fi_a*w*Cf
    B44F0 = 0.787*ra*Sf*r_f**2*sqrt(w*visc)*(1 + 0.00814*(r_f**2*fi_a**2*w/visc)**0.386)
    B44F= B44F0*(1 + 4.1*V/(w*L))

    return B44F

def hull_lift(V,B, d, OG, L, A, ra=1000):
    """
    Calculate the hull lift force damping

    Journee, J.M.J., Adegeest, L.J.M.,
    2003. Theoretical Manual of Strip Theory Program &quot;
    Seaway for Windows&quot; TU Delft, Report 1370.


    Parameters
    ----------
    V
        Ship speed [m/s]
    B
        Ship beam [m]
    d
        Ship draught [m]
    OG
        Distance from roll axis to still water level [m]
    ra
        density of water [kg/m3]
    L
        Ship length [m]
    A
        Mid section area [m2]

    Returns
    -------
    B44L
        Hull lift damping [Nm]

    """

    lo   = 0.3*d;
    lR   = 0.5*d;
    #K    = 0.1;         #!! depends on CM  # S175
    C_mid = A/(B*d)
    K = 106 * (C_mid - 0.91)**2 - 700 * (C_mid - 0.91)**3;  # Fit by Journee

    kN   = 2*pi*d/L+K*(4.1*B/L-0.045);
    #B44L = rho/2*V*L*d*kN*lo*lR*(1+1.4*OG/lR+0.7*OG**2/(lo*lR))  # S175
    B44L = ra / 2 * V * L * d * kN * lo * lR * (1 - 1.4 * (OG) / lR + 0.7 * OG**2 / (lo * lR));

    return B44L

def calculate_B44(w, V, d, Bw0, fi_a,  B,  A, bBK, R, OG, Ho, ra, Cb, L, LBK, visc =   1.15*10**-6, g=9.81, Bw_div_Bw0_max=12):
    """

    Parameters
    ----------
    w : float
        roll frequency [rad/s]
    V : float
        Ship speed [m/s]
    d : float
        Draught of hull [m]
    Bw0 : float
        Zero speed wave damping [Nm*s/rad]
    fi_a : float
        Roll amplitude [rad]
    B : float
        Breadth of hull [m]
    A : float
        Area of cross section of hull [m2]
    bBK : float
        breadth of Bilge keel [m] !!(=height???)
    R : float
        Bilge Radis [m]
    OG : float
        distance from roll axis to still water level [m]
    Ho : float
        half breadth to draft ratio B/(2*d) [-]
    ra : float
         density of water (1025) [kg/m3]
    Cb : float
        Block coeff [-]
    L : float
        Ship length [m]
    LBK : float
        Bilge keel length [m]
    visc : float
        kinematic viscosity [m2/s]
    g : float
        gravity accelaration [m/s2]
    Bw_div_Bw0_max : float
        maxmum allowed difference between Bw0 and at speed.

    Returns
    -------
    B44 : float
        Total damping [Nm*s/rad]
    BW44 : float
        Wave damping [Nm*s/rad]
    B44_BK : float
        Bilge keel damping [Nm*s/rad]
    B44F : float
        Friction damping [Nm*s/rad]
    B44L : float
        Hull lift damping [Nm*s/rad]

    """

    BW44=Bw(w, V, d, Bw0, g=9.81, Bw_div_Bw0_max=Bw_div_Bw0_max)

    Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0 = bilge_keel(w, fi_a, V, B, d, A, bBK, R, g, OG, Ho, ra)
    B44BK_N0 = Bp44BK_N0*LBK
    B44BK_H0 = Bp44BK_H0*LBK
    B44BK_L = B44BK_L
    # B44BKW0 = B44BKW0 * dim...
    B44_BK = B44BK_N0 + B44BK_H0 + B44BK_L

    B44F = frictional(w,fi_a,V,B, d, OG, ra, Cb, L, visc)

    B44L = hull_lift(V,B, d, OG, L, A, ra)

    B44 = BW44+B44_BK+B44F+B44L

    return B44,BW44,B44_BK,B44F,B44L

def calculate_B44_series(row, Bw_div_Bw0_max=12):
    s = pd.Series(name=row.name)
    B_44,B_W,B_BK,B_F,B_L = calculate_B44(w=row['w'], V=row['V'], d=row['d'], Bw0=row['Bw0'], fi_a=row['fi_a'], B=row['B'], A=row['A'], bBK=row['BKB'], R=row['R_b'],
                         OG=row['OG'], Ho=row['Ho'], ra=row['rho'], Cb=row['Cb'], L=row['L'], LBK=row['LBK'], visc=row['visc'], g=row['g'], Bw_div_Bw0_max=Bw_div_Bw0_max)
    s['B_44']=B_44
    s['B_W']=B_W
    s['B_BK']=B_BK
    s['B_F']=B_F
    s['B_L']=B_L
    return s


def calculate_sectional_lewis(B_s:np.ndarray, T_s:np.ndarray, S_s:np.ndarray):
    """
    Lewis form approximation' is obtained.
    Given the section's area, S_s, beam B and draught T, the constants a, a a_3 are uniquely defined
    by von Kerczek and Tuck18 as:

    Parameters
    ----------
    B_s : array_like
        Sectional beams [m]
    T_s : array_like
        Sectional draughts [m]
    S_s : array_like
        Sectional area [m2]

    Returns
    -------
    a, a_1, a_3, sigma_s, H : array_like
        sectional lewis coefficients.

    """
    B_s=copy(B_s)
    T_s=copy(T_s)
    S_s=copy(S_s)

    if isinstance(B_s, int):
        B_s = float(B_s)

    if isinstance(B_s,float):
        B_s = np.array([B_s])

    if isinstance(T_s, int):
        T_s = float(T_s)

    if isinstance(T_s,float):
        T_s = np.array([T_s])

    if isinstance(S_s, int):
        S_s = float(S_s)

    if isinstance(S_s,float):
        S_s = np.array([S_s])

    #H = B_s / (2 * T_s)
    H = np.divide(B_s, (2 * T_s), out=np.zeros_like(B_s), where=(2 * T_s)!=0)

    #sigma_s = S_s / (B_s * T_s)
    sigma_s = np.divide(S_s, (B_s * T_s), out=np.zeros_like(S_s), where=(B_s * T_s)!=0)

    C_1 = (3 + 4 * sigma_s / np.pi) + (1 - 4 * sigma_s / np.pi) * ((H - 1) / (H + 1)) ** 2
    a_3 = (-C_1 + 3 + np.sqrt(9 - 2 * C_1)) / C_1
    a_1 = (1 + a_3) * (H - 1) / (H + 1)
    #a = B_s / (2 * (1 + a_1 + a_3))
    a = np.divide(B_s, (2 * (1 + a_1 + a_3)), out=np.zeros_like(B_s), where=(2 * (1 + a_1 + a_3))!=0)

    return a, a_1, a_3, sigma_s, H

def eddy(bwl:np.ndarray, a_1:np.ndarray, a_3:np.ndarray, sigma:np.ndarray, xs:np.ndarray, H0:np.ndarray, Ts:np.ndarray,
         OG:float, R:float, wE:float, fi_a:float, ra=1000.0):
    """
        Calculation of eddy damping according to Ikeda.
        This implementation is a translation from Carl-Johans Matlab implementation.

        Parameters
        ----------
        bwl
            sectional beam water line [m]
        a_1
            sectional lewis coefficients
        a_3
            sectional lewis coefficients
        sigma
            sectional coefficient
        xs
            sectional x position [m]
        H0
            sectional coefficient
        Ts
            sectional draft [m]
        OG
            vertical distance water line to cg [m]
        R
            bilge radius [m]
        ra
            water density [kg/m3]
        wE
            roll requency [rad/s]
        fi_a
            roll amplitude [rad]

        Returns
        -------
        B_E0
            Eddy damping at zero speed.
        """
    Bp44E0s = eddy_sections(bwl=bwl, a_1=a_1, a_3=a_3, sigma=sigma, H0=H0, Ts=Ts, OG=OG, R=R, wE=wE, fi_a=fi_a, ra=ra)
    Bp44E0 = simps(y=Bp44E0s, x=xs, axis=0)

    return Bp44E0

def eddy_sections(bwl:np.ndarray, a_1:np.ndarray, a_3:np.ndarray, sigma:np.ndarray, H0:np.ndarray, Ts:np.ndarray,
         OG:float, R:float, wE:float, fi_a:float, ra=1000.0):
    """
    Calculation of eddy damping according to Ikeda.
    This implementation is a translation from Carl-Johans Matlab implementation.

    Parameters
    ----------
    bwl
        sectional beam water line [m]
    a_1
        sectional lewis coefficients
    a_3
        sectional lewis coefficients
    sigma
        sectional coefficient
    H0
        sectional coefficient
    Ts
        sectional draft [m]
    OG
        vertical distance water line to cg [m]
    R
        bilge radius [m]
    ra
        water density [kg/m3]
    wE
        roll requency [rad/s]
    fi_a
        roll amplitude [rad]

    Returns
    -------
    Bp44E0s
        Eddy damping per unit length for the sections at zero speed.
    """

    #H0=np.array(H0)/2  # ...strange...
    H0 = np.array(H0) # ...strange...

    N=len(bwl)
    #M = bwl / (2 * (1 + a_1 + a_3));
    M = np.divide(bwl, (2 * (1 + a_1 + a_3)), out=np.zeros_like(bwl), where=(2 * (1 + a_1 + a_3))!=0)

    fi1 = 0;
    #fi2 = 0.5 * arccos(a_1 * (1 + a_3)) / (4 * a_3);
    fi2 = np.divide(0.5 * arccos(a_1 * (1 + a_3)), (4 * a_3), out=np.zeros_like(a_1), where=(4 * a_3)!=0)
    rmax_fi1 = M*sqrt(((1+a_1)*sin(fi1)-a_3*sin(fi1))**2+((1-a_1)*cos(fi1)-a_3*cos(fi1))**2)
    rmax_fi2 = M*sqrt(((1+a_1)*sin(fi2)-a_3*sin(fi2))**2+((1-a_1)*cos(fi2)-a_3*cos(fi2))**2)

    mask=rmax_fi2 > rmax_fi1
    fi=np.zeros(N)
    fi[mask] = fi2[mask]
    fi[~mask] = fi1

    B0 = -2 * a_3 * sin(5 * fi) + a_1 * (1 - a_3) * sin(3 * fi) + (
                (6 + 3 * a_1) * a_3 ** 2 + (3 * a_1 + a_1 ** 2) * a_3 + a_1 ** 2) * sin(fi)

    A0 = -2 * a_3 * cos(5 * fi) + a_1 * (1 - a_3) * cos(3 * fi) + (
                (6 - 3 * a_1) * a_3 ** 2 + (a_1 ** 2 - 3 * a_1) * a_3 + a_1 ** 2) * cos(fi)

    H = 1 + a_1 ** 2 + 9 * a_3 ** 2 + 2 * a_1 * (1 - 3 * a_3) * cos(2 * fi) - 6 * a_3 * cos(4 * fi)

    f3 = 1 + 4 * exp(-1.65 * 10 ** 5 * (1 - sigma) ** 2);

    x_ = np.array([rmax_fi1, rmax_fi2]).transpose()
    rmax = max(x_, axis=1)

    H0_prim = H0*Ts/(Ts-OG)
    sigma_prim = (sigma*Ts-OG)/(Ts-OG)
    gamma=sqrt(pi)*f3*(rmax+2*M/H*sqrt(B0**2*A0**2))/((2*Ts*(1-OG/Ts)*sqrt(H0_prim*sigma_prim)))

    f1 = 0.5 * (1 + tanh(20 * (sigma - 0.7)));
    f2 = 0.5 * (1 - cos(pi * sigma)) - 1.5 * (1 - exp(-5 * (1 - sigma))) * (sin(pi * sigma)) ** 2

    Cp = 0.5 * (0.87 * exp(-gamma) - 4 * exp(-0.187 * gamma) + 3);

    M_re = 1 / 2 * ra * rmax ** 2 * Ts ** 2 * Cp * (
                (1 - f1 * R / Ts) * (1 - OG / Ts - f1 * R / Ts) + f2 * (H0 - f1 * R / Ts) ** 2)
    Cr = M_re / (1 / 2 * ra * Ts ** 4)


    #WE, CR = np.meshgrid(wE, Cr)
    Cr=np.array(Cr)
    wE=np.array(wE)

    try:
        len_Cr = len(Cr)
    except TypeError:
        len_Cr = 1

    try:
        len_wE = len(wE)
    except TypeError:
        len_wE = 1

    WE=np.tile(wE, (len_Cr, 1))
    FI_a=np.tile(fi_a, (len_Cr, 1))

    Tss = np.tile(Ts, (len_wE, 1)).transpose()
    CR = np.tile(Cr, (len_wE, 1)).transpose()

    Bp44E0s = 4 * ra * Tss ** 4 * WE * FI_a * CR / (3 * pi)

    mask = np.isnan(Bp44E0s)
    Bp44E0s[mask] = 0

    return Bp44E0s








