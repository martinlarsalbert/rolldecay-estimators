from numpy import sqrt,pi,tanh, exp, cos, sin, arccos
import numpy as np

def eddy_sections(bwl:np.ndarray, a_1:np.ndarray, a_3:np.ndarray, sigma:np.ndarray, H0:np.ndarray, Ts:np.ndarray,
         OG:float, R:float, wE:float, fi_a:float, ra=1000.0):
    """
    Calculation of eddy damping according to
    Ikeda, Y.,
    1978. On eddy making component of roll damping force on naked hull. University of Osaka Prefacture,
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
    B_e0s
        Eddy damping per unit length for the sections at zero speed.
    """

    d = Ts

    C_r = calculate_C_r(bwl=bwl,a_1=a_1, a_3=a_3, sigma=sigma, H0=H0, d=Ts, OG=OG, R=R, ra=ra)

    B_e0s = 4*ra/(3*pi)*d**4*wE*fi_a*C_r  # (6) (Rewritten)

    return np.array([B_e0s])

def calculate_C_r(bwl:np.ndarray, a_1:np.ndarray, a_3:np.ndarray, sigma:np.ndarray, H0:np.ndarray, d,
         OG:float, R:float, ra=1000.0):

    """
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

    Returns
    -------
    B_e0s
        Eddy damping per unit length for the sections at zero speed.
    """


    ## start to obtain y from the parameter H0 and a of the cylinder section:
    gamma, r_max = calculate_gamma(sigma=sigma, OG=OG, d=d, a_1=a_1, a_3=a_3, H0=H0, bwl=bwl)

    C_p = calculate_C_p(gamma=gamma)  # (14)

    f_1 = 1/2*(1 + tanh(20*(sigma - 0.7)))  # (12)
    f_2 = calculate_f2(sigma=sigma)  # (15)

    P_m_ = 3*1/2*ra*r_max**2*C_p  # (13) (*|phi1d|*phi2d left out, but will cancel out in the next)
    C_r = 2/(ra*d**2)*((1 - f_1*R/d)*(1 - OG/d - f_1*R/d) + f_2*(H0 - f_1*R/d)**2)*P_m_/3  # (10)
    
    #M_re = 1/2*ra*r_max**2*d**2*C_p*((1-f_1*R/d)*(1 - OG/d - f_1*R/d) + f_2*(H0 - f_1*R/d)**2)
    #C_r = M_re/(1/2*ra*d**4)
    
    return C_r


def calculate_f2(sigma):
    f_2 = 1 / 2 * (1 - cos(pi * sigma)) - 1.5 * (1 - exp(-5 * (1 - sigma))) * sin(pi * sigma) ** 2  # (15)
    return f_2

def calculate_C_p(gamma):
    C_p = 1 / 2 * (0.87 * exp(-gamma) - 4 * exp(-0.187 * gamma) + 3)  # (14)
    return C_p

def calculate_gamma(sigma, OG, d, a_1, a_3, H0, bwl):

    # (A-1):
    sigma_ = (sigma-OG/d)/(1-OG/d)
    H0_ = H0/(1-OG/d)

    # (A-4)
    def calculate_A(psi):
        return -2 * a_3 * cos(5 * psi) + a_1 * (1 - a_3) * cos(3 * psi) + (
            (6 - 3 * a_1) * a_3 ** 2 + (a_1 ** 2 - 3 * a_1) * a_3 + a_1 ** 2) * cos(psi)

    def calculate_B(psi):
        return -2 * a_3 * sin(5 * psi) + a_1 * (1 - a_3) * sin(3 * psi) + (
            (6 + 3 * a_1) * a_3 ** 2 + (3 * a_1 + a_1 ** 2) * a_3 + a_1 ** 2) * sin(psi)

    def calculate_M():
        return bwl / (2 * (1 + a_1 + a_3))  # Note B is bwl!!!

    def calculate_H(psi):
        return 1 + a_1 ** 2 + 9 * a_3 ** 2 + 2 * a_1 * (1 - 3 * a_3) * cos(2 * psi) - 6 * a_3 * cos(4 * psi)

    def calculate_r_max(psi):

        M = calculate_M()

        return M * sqrt(
            ((1 + a_1) * sin(psi) - a_3 * sin(3 * psi)) ** 2 + (
                        (1 - a_1) * cos(psi) + a_3 * cos(3 * psi)) ** 2)  # (A-5)

    # (A-6)
    psi_1 = 0
    factor = a_1*(1+a_3)/(4*a_3)  
    # Limit factor to [-1,1]
    mask = np.abs(factor) > 1
    factor[mask] = 1*np.sign(factor[mask])
    psi_2 = 1/2*arccos(factor)

    # (A-7)
    r_max_1 = calculate_r_max(psi_1)
    r_max_2 = calculate_r_max(psi_2)
    mask = r_max_1 >= r_max_2
    psi = np.array(psi_2)
    psi[mask] = psi_1

    A = calculate_A(psi=psi)
    B = calculate_B(psi=psi)
    M = calculate_M()
    H = calculate_H(psi=psi)

    r_max = np.array(r_max_2)
    r_max[mask] = r_max_1[mask]

    f_3 = 1 + 4*exp(-1.65*10**5*(1-sigma)**2)  # (A-9)

    gamma = sqrt(pi)*f_3/((2*d)*(1-OG/d)*sqrt(H0_*sigma_))*(r_max+2*M/H)*sqrt(A**2+B**2)  # (A-10)

    return gamma, r_max
