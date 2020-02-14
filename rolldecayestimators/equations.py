import sympy as sp
from rolldecayestimators.symbols import *

##### Quadratic
lhs = phi_dot_dot + 2*zeta*omega0*phi_dot + d*sp.Abs(phi_dot)*phi_dot + omega0**2*phi
roll_diff_equation = sp.Eq(lhs=lhs,rhs=0)

# Solve the diff equation by introducing helper variables:
phi_old,p_old = me.dynamicsymbols('phi_old p_old')
velocity_equation = sp.Eq(lhs=phi.diff(),rhs=p_old)
roll_diff_equation_subs = roll_diff_equation.subs(
    [
        (phi.diff(), p_old),
        (phi, phi_old),

    ]
)
solution = sp.solve(roll_diff_equation_subs,(p_old.diff()))[0]
acceleration_equation = sp.Eq(lhs=phi.diff().diff(), rhs=solution)


##### Linear
lhs = phi_dot_dot + 2*zeta*omega0*phi_dot + omega0**2*phi
roll_diff_equation_linear = sp.Eq(lhs=lhs,rhs=0)

# Solve the diff equation by introducing helper variables:
phi_old,p_old = me.dynamicsymbols('phi_old p_old')
velocity_equation_linear = sp.Eq(lhs=phi.diff(),rhs=p_old)
roll_diff_equation_linear_subs = roll_diff_equation_linear.subs(
    [
        (phi.diff(), p_old),
        (phi, phi_old),

    ]
)
solution = sp.solve(roll_diff_equation_linear_subs,(p_old.diff()))[0]
acceleration_equation_linear = sp.Eq(lhs=phi.diff().diff(), rhs=solution)