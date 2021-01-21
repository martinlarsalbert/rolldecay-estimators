import numpy as np
import rolldecayestimators.lambdas as lambdas

def eddy(rho, lpp, d, beam, OG, phi_a, w):
    """
    Calculate eddy damping for a barge shaped hull accoring to:

    Ikeda, Y., 1993. Roll Damping of a Sharp-Cornered Barge and Roll Control by New Type Stabilizer,
    in: Proceedings of the 3rd International Offshore and Polar Engineering Conference.
    Presented at the The International Society of Offshore and Polar Engineers, Singapore.

    Parameters
    ----------
    rho
        water density [kg/m3]
    lpp
        ship length [m]
    d
        draught [m]
    beam
        ship beam [m]
    OG
         distance from the still water level O to the roll axis G [m]
    phi_a
        roll amplitude [rad]
    w
        frequency of roll motion [rad/s]
    Returns
    -------
    B_E
    """

    H_0 = beam/(2*d)
    B_E = (2/np.pi)*rho*lpp*d**4*(H_0**2 + 1 - OG/d)*(H_0**2 + (1 - OG/d)**2)*phi_a*w
    return B_E