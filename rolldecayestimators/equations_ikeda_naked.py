
"""
Equations to predict eddy damping according to:

[1]
Ikeda, Y.,
1978. On eddy making component of roll damping force on naked hull. University of Osaka Prefacture,
Department of Naval Architecture, Japan, Report No. 00403,
Published in: Journal of Society of Naval Architects of Japan, Volume 142.

"""

import sympy as sp
from rolldecayestimators.symbols import *
from sympy import pi

eq_B_E_star_hat = sp.Eq(B_E0_hat,
                        8/(3*pi)*B_E_star_hat  # (6) (part 1)
                        )

eq_B_E0_hat = sp.Eq(B_E0_hat,
                    4*L_pp*T**4*omega_hat*phi_a/(3*pi*Disp*beam**2)*C_r  # (6) (part 2)
                    )

eq_C_r = sp.Eq(C_r,
               2/(rho*T**2)*((1 - f_1*R_b/T)*(1 - OG/T - f_1*R_b/T) + f_2*(H_0 - f_1*R_b/T)**2)*P_m/3  # (10)
               )

eq_P_m = sp.Eq(P_m,
               3*1/2*rho*r_max**2*C_P*sp.Abs(phi.diff())*phi.diff().diff()  # (13)
              )

