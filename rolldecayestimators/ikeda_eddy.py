import numpy as np
from numpy import pi, abs, tanh, cos, sin, exp, sqrt, arccos

def calculate_rmax(M, a_1, a_3, psi):
    r_maxs = M * sqrt(
        ((1 + a_1) * sin(psi) - a_3 * sin(3 * psi)) ** 2 + ((1 - a_1) * cos(psi) + a_3 * cos(3 * psi)) ** 2)
    return r_maxs

def eddy_sections(bwl:np.ndarray, a_1:np.ndarray, a_3:np.ndarray, sigma:np.ndarray, H0:np.ndarray, Ts:np.ndarray,
                  OG:float, R_b:float, wE:float, fi_a:float, rho=1000.0):
    """
    Calculation of eddy damping according to Ikeda.
    This implementation is a translation from:
    Ikeda, Y., 1978. On eddy making component of roll damping force on naked hull. University of Osaka Prefacture,
    Department of Naval Architecture, Japan, Report No. 00403,
    Published in: Journal of Society of Naval Architects of Japan, Volume 142.

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
    R_b
        bilge radius [m]
    rho
        water density [kg/m3]
    wE
        roll requency [rad/s]
    fi_a
        roll amplitude [rad]

    Returns
    -------
    BE0s_hat
        Eddy damping per unit length for the sections at zero speed.
    """

    d = Ts

    B = bwl  # Probably
    M = B / (1 + a_1 + a_3)

    f_1 = 1/2*(1+tanh(20*(sigma-0.7)))
    f_2 = 1/2*(1-cos(pi*sigma))-1.5*(1-exp(-5*(1-sigma)))*sin(pi*sigma)**2
    H0_prim = H0*d/(d-OG)
    sigma_prim = (sigma*d-OG)/(d-OG)
    f_3 = 1 + 4*exp(-1.65*10**5*(1-sigma)**2)

    psi_1 = 0.0
    psi_2 = 1 / 2 * arccos(a_1 * (1 + a_3) / (4 * a_3))

    r_max_1 = calculate_rmax(M=M, a_1=a_1, a_3=a_3, psi=psi_1)
    r_max_2 = calculate_rmax(M=M, a_1=a_1, a_3=a_3, psi=psi_2)

    mask = r_max_1 >= r_max_2
    psi = psi_2
    psi[mask] = psi_1

    ## A
    cs = [-2 * a_3, a_1*(1 - a_3), ((6 - 3*a_1)*a_3**2 + (a_1**2 - 3*a_1)*a_3 + a_1**2)]
    ps = [5*psi, 3*psi, psi]
    A_1=0
    for c,p in zip(cs,ps):
        A_1+=c*cos(p)

    ## B
    B_1 = 0
    cs = [-2 * a_3, a_1 * (1 - a_3), ((6 + 3 * a_1)*a_3**2 + (3*a_1 + a_1**2)*a_3 + a_1**2)]
    for c, p in zip(cs, ps):
        B_1 += c * sin(p)


    r_max = calculate_rmax(M=M, a_1=a_1, a_3=a_3, psi=psi)

    H = 1 + a_1**2 + 9*a_3**2 + 2*a_1*(1-3*a_3)*cos(2*psi) - 6*a_3*cos(4*psi)
    V_max_div_phi1d = f_3*(r_max + 2*M/H*sqrt(A_1**2+B_1**2))
    gamma = sqrt(pi)/(2*d*(1-OG/d)*sqrt(H0_prim*sigma_prim))*V_max_div_phi1d

    C_p = 1/2*(0.87*exp(-gamma)-4*exp(-0.187*gamma)+3)


    R_b = calculate_R_b(beam=bwl, draught=Ts, H0=H0, sigma=sigma)

    M_re = 1/2*rho*r_max**2*d**2*C_p*((1-f_1*R_b/d)*(1-OG/d-f_1*R_b/d) + f_2*(H0-f_1*R_b/d)**2)
    C_r = M_re / (1/2*rho*d**4)
    BE0s = 4/(3*pi)*d**4*wE*fi_a*C_r  # Rewritten without hat
    return BE0s

def calculate_R_b(beam, draught, H0, sigma):
    """
    Calculate bilge radius with Ikedas empirical formula:
    Returns
    -------
    R_b : ndarray
        Bilge radius [m]

    """

    mask=sigma>1

    sigma[mask]=0.99  # Do avoid negative value in sqrt
    mask=H0<0
    R_b = 2*draught*np.sqrt(H0*(sigma-1)/(pi-4))

    mask = (H0>=1) & (R_b/draught>1)
    R_b[mask]=draught

    mask = (H0 < 1) & (R_b / draught > H0)
    R_b[mask] = beam/2

    return R_b