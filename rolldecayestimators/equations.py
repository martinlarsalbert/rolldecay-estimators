import sympy as sp
from rolldecayestimators.symbols import *
import rolldecayestimators.direct_linear_estimator as direct_linear_estimator

# General roll motion equation according to Himeno:
lhs = A_44*phi_dot_dot + B_44 + C_44
rhs = M_44
roll_equation_himeno = sp.Eq(lhs=lhs, rhs=rhs)


# General equation for roll motion:
lhs = I*phi_dot_dot + B*phi_dot + m*g*GZ
rhs = M_x
roll_equation_general = sp.Eq(lhs=lhs, rhs=rhs)

# No external forces (during roll decay)

roll_decay_equation_general = roll_equation_general.subs(M_x,0)

##### Linear
linear_stiffness_equation = sp.Eq(GZ,GM*phi)

subs = [
    (GZ, sp.solve(linear_stiffness_equation, GZ)[0]),
]
roll_decay_linear_equation = roll_decay_equation_general.subs(subs)

linear_damping_equation = sp.Eq(B/I,zeta)


# Solve the diff equation by introducing helper variables:
phi_old,p_old = me.dynamicsymbols('phi_old p_old')
velocity_equation_linear = sp.Eq(lhs=phi.diff(),rhs=p_old)
roll_diff_equation_linear_subs = direct_linear_estimator.roll_diff_equation.subs(
    [
        (phi.diff(), p_old),
        (phi, phi_old),

    ]
)
solution = sp.solve(roll_diff_equation_linear_subs,(p_old.diff()))[0]
acceleration_equation_linear = sp.Eq(lhs=phi.diff().diff(), rhs=solution)

A,B = sp.symbols('A B')  # helpers
A_equation = sp.Eq(lhs=A, rhs=-2*zeta/omega0)
B_equation = sp.Eq(lhs=B, rhs=-1/(omega0**2))

zeta_equation = sp.Eq(lhs=zeta, rhs=sp.solve(A_equation,zeta)[0])
omega_equation = sp.Eq(lhs=omega0, rhs=sp.solve(B_equation,omega0)[0])

##### Quadratic


C = sp.symbols('C')  # helpers
C_equation = sp.Eq(lhs=C, rhs=B*d)
d_equation = sp.Eq(lhs=d, rhs=sp.solve(C_equation,d)[0])

##### Quadratic improved
quadratic_stiffness_equation = sp.Eq(GZ,GM*phi+dGM*sp.Abs(phi)*phi)