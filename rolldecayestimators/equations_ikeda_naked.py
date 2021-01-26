
"""
Equations to predict eddy damping according to:

[1]
Ikeda, Y.,
1978. On eddy making component of roll damping force on naked hull. University of Osaka Prefacture,
Department of Naval Architecture, Japan, Report No. 00403,
Published in: Journal of Society of Naval Architects of Japan, Volume 142.

[2]
ITTC, 2011. ITTC – Recommended Procedures Numerical Estimation of Roll Damping.


"""

import sympy as sp
from rolldecayestimators.symbols import *
from sympy import pi,sqrt

eq_B_E_star_hat = sp.Eq(B_E0_hat,
                        8/(3*pi)*B_E_star_hat  # (6) (part 1)
                        )

eq_B_E0_hat = sp.Eq(B_E0_hat,
                    4*L_pp*T**4*omega_hat*phi_a/(3*pi*Disp*beam**2)*C_r  # (6) (part 2)
                    )
eq_volume = sp.Eq(Disp,
                  T_s*B_s*sigma*L_pp,
                  )
solution = sp.solve([eq_B_E0_hat, eq_volume, eq_B_E_star_hat],
                    C_r,Disp,B_E_star_hat,
                    dict=True
                    )
eq_C_r_2 = sp.Eq(C_r, solution[0][C_r])

eq_C_r = sp.Eq(C_r,
               2/(rho*T**2)*((1 - f_1*R_b/T)*(1 - OG/T - f_1*R_b/T) + f_2*(H_0 - f_1*R_b/T)**2)*P_m/3  # (10)
               )

eq_P_m = sp.Eq(P_m,
               3*1/2*rho*r_max**2*C_P*sp.Abs(phi.diff())*phi.diff().diff()  # (13)
              )

eq_B_E0s = sp.Eq(B_E0s,
                 4 * rho * T_s**4*omega*phi_a * C_r / (3 * pi) # (2.19) [2]
            )

x, AP, FP = sp.symbols('x AP FP')
eq_B_E0 = sp.Eq(B_E0,
                sp.Integral(B_E0s,(x,AP,FP))
     )

eq_R_b = sp.Eq(R_b,
               sqrt((B_s*T_s*(1-sigma))/(1-pi/4))  # Martin R_b approximations
               )

## Lewis:
eq_D_1 = sp.Eq(D_1,
                (3 + 4 * sigma / pi) + (1 - 4 * sigma / pi) * ((H_0 - 1) / (H_0 + 1)) ** 2
                )

eq_a_3 = sp.Eq(a_3,
               (-D_1 + 3 + sqrt(9 - 2 * D_1)) / D_1
               )
eq_a_1 = sp.Eq(a_1,
               (1 + a_3) * (H_0 - 1) / (H_0 + 1)
               )