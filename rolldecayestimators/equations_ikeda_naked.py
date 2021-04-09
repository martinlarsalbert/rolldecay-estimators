
"""
Equations to predict eddy damping according to:

[1]
Ikeda, Y.,
1978. On eddy making component of roll damping force on naked hull. University of Osaka Prefacture,
Department of Naval Architecture, Japan, Report No. 00403,
Published in: Journal of Society of Naval Architects of Japan, Volume 142.

[2]
ITTC, 2011. ITTC – Recommended Procedures Numerical Estimation of Roll Damping.

[3]
Ikeda, Y., Himeno, Y., Tanaka, N., 1978. 
Components of Roll Damping of Ship at Forward Speed. 
J. SNAJ, Nihon zousen gakkai ronbunshu 1978, 113–125. https://doi.org/10.2534/jjasnaoe1968.1978.113


"""

import sympy as sp
from rolldecayestimators.symbols import *
from sympy import pi,sqrt
from rolldecayestimators import equations
from rolldecayestimators import symbols

eq_B_E_star_hat = sp.Eq(B_E0_hat,
                        8/(3*pi)*B_E_star_hat  # (6) (part 1)
                        )

eq_B_E0_hat = sp.Eq(B_E0_hat,
                    4 * L_pp * T_s ** 4 * omega_hat * phi_a / (3 * pi * Disp * b ** 2) * C_r  # (6) (part 2)
                    )
eq_volume = sp.Eq(Disp,
                  T_s*B_s*sigma*L_pp,
                  )
solution = sp.solve([eq_B_E0_hat, eq_volume, eq_B_E_star_hat],
                    C_r,Disp,B_E_star_hat,
                    dict=True
                    )
eq_C_r_2 = sp.Eq(C_r, solution[0][C_r])

eqs = [eq_C_r_2,
 equations.omega_hat_equation]
eq_C_r_omega =sp.Eq(symbols.C_r,
      sp.solve(eqs, symbols.C_r, symbols.omega_hat, dict=True)[0][symbols.C_r])

eq_C_r = sp.Eq(C_r,
               2/(rho*T_s**2)*((1 - f_1*R_b/T_s)*(1 - OG/T_s - f_1*R_b/T_s) + f_2*(H_0 - f_1*R_b/T_s)**2)*P_m/3  # (10)
               )

eq_P_m = sp.Eq(P_m,
               3*1/2*rho*r_max**2*C_P*sp.Abs(phi.diff())*phi.diff().diff()  # (13)
              )

eq_B_E0s = sp.Eq(B_E0s,
                 4 * rho * T_s**4*omega*phi_a * C_r / (3 * pi) # (2.19) [2]
            )

eq_B_E0 = sp.Eq(B_E0,
                sp.Integral(B_E0s,(x_s,AP,FP))
     )

eq_R_b = sp.Eq(R_b,
               sqrt((B_s*T_s*(1-sigma))/(1-pi/4))  # Martin R_b approximations
               )


eq_B_star_hat = sp.Eq(B_star_hat,
                      B_E_star_hat +  B_W_star_hat + B_F_star_hat)

## Speed dependancy
eq_K = sp.Eq(K,
             L_pp*omega/V
             )


eq_eddy_speed_general = sp.Eq(B_E/B_E0,
                     (alpha*K)**2 / ((alpha*K)**2 + 1) 
                     )

eq_eddy_speed = eq_eddy_speed_general.subs(alpha, 0.04)  ## alpha from Ikeda paper.

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