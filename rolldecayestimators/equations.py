import sympy as sp
from rolldecayestimators.symbols import *

# General roll motion equation according to Himeno:
lhs = A_44*phi_dot_dot + B_44 + C_44
rhs = M_44
roll_equation_himeno = sp.Eq(lhs=lhs, rhs=rhs)

# No external forces (during roll decay)
roll_decay_equation_general_himeno = roll_equation_himeno.subs(M_44,0)


restoring_equation = sp.Eq(C_44,m*g*GZ)
restoring_equation_linear = sp.Eq(C_44,m*g*GM*phi)
restoring_equation_quadratic = sp.Eq(C_44, C_1 * phi + C_3 * phi ** 3)
restoring_equation_cubic = sp.Eq(C_44,C_1*phi + C_3*phi*sp.Abs(phi) + C_5*phi**3)

## Cubic model:
b44_cubic_equation = sp.Eq(B_44, B_1 * phi_dot + B_2 * phi_dot * sp.Abs(phi_dot) + B_3 * phi_dot ** 3)
restoring_equation_cubic = sp.Eq(C_44, C_1 * phi + C_3 * phi ** 3 + C_5 * phi ** 5)

subs = [
    (B_44, sp.solve(b44_cubic_equation, B_44)[0]),
    (C_44, sp.solve(restoring_equation_cubic, C_44)[0])
]
roll_decay_equation_himeno_quadratic = roll_decay_equation_general_himeno.subs(subs)


## Quadratic model:
b44_quadratic_equation = sp.Eq(B_44, B_1 * phi_dot + B_2 * phi_dot * sp.Abs(phi_dot))
restoring_equation_quadratic = sp.Eq(C_44, C_1 * phi + C_3 * phi ** 3)

subs = [
    (B_44, sp.solve(b44_quadratic_equation, B_44)[0]),
    (C_44, sp.solve(restoring_equation_quadratic, C_44)[0])
]
roll_decay_equation_himeno_quadratic = roll_decay_equation_general_himeno.subs(subs)

## Linear model:
b44_linear_equation = sp.Eq(B_44, B_1 * phi_dot)
restoring_linear_quadratic = sp.Eq(C_44, C_1 * phi)

subs = [
    (B_44, sp.solve(b44_linear_equation, B_44)[0]),
    (C_44, sp.solve(restoring_linear_quadratic, C_44)[0])
]
roll_decay_equation_himeno_linear = roll_decay_equation_general_himeno.subs(subs)

C_equation = sp.Eq(C,C_44/phi)
C_equation_linear = C_equation.subs(C_44,sp.solve(restoring_equation_linear,C_44)[0])
C_equation_cubic = C_equation.subs(C_44,sp.solve(restoring_equation_cubic,C_44)[0])
C_equation_quadratic = C_equation.subs(C_44,sp.solve(restoring_equation_quadratic,C_44)[0])

roll_decay_equation_himeno_quadratic =  roll_decay_equation_general_himeno.subs(B_44,
                                                        sp.solve(b44_quadratic_equation,B_44)[0]).subs(C,
                                                            sp.solve(C_equation_quadratic,C)[0])

subs = [
    (B_44, sp.solve(b44_quadratic_equation, B_44)[0]),
    (C_44, sp.solve(restoring_linear_quadratic, C_44)[0])
]
roll_decay_equation_himeno_quadratic_b = roll_decay_equation_general_himeno.subs(subs)


roll_decay_equation_himeno_quadratic_c = roll_decay_equation_himeno_quadratic.subs(C_44,sp.solve(C_equation, C_44)[0])
zeta_equation = sp.Eq(2*zeta*omega0,B_1/A_44)
d_equation = sp.Eq(d,B_2/A_44)
omega0_equation = sp.Eq(omega0,sp.sqrt(C/A_44))

eq = sp.Eq(roll_decay_equation_himeno_quadratic_c.lhs/A_44,0)  # helper equation

subs = [
    (B_1, sp.solve(zeta_equation, B_1)[0]),
    (B_2, sp.solve(d_equation, B_2)[0]),
    (C / A_44, sp.solve(omega0_equation, C / A_44)[0])

]

roll_decay_equation_quadratic = sp.Eq(sp.expand(eq.lhs).subs(subs), 0)
roll_decay_equation_quadratic = sp.factor(roll_decay_equation_quadratic, phi_dot)
roll_decay_equation_linear = roll_decay_equation_quadratic.subs(d,0)

omega0_equation_linear = omega0_equation.subs(C,sp.solve(C_equation_linear,C)[0])
A44 = sp.solve(omega0_equation_linear, A_44)[0]
zeta_B1_equation = zeta_equation.subs(A_44,A44)
d_B2_equation = d_equation.subs(A_44,A44)

## Cubic model:
subs = [
    (B_44,sp.solve(b44_cubic_equation,B_44)[0]),
    (C_44,sp.solve(restoring_equation_cubic,C_44)[0])
]
roll_decay_equation_cubic = roll_decay_equation_general_himeno.subs(subs)
# But this equation does not have a unique solution, so we devide all witht the interia A_44:

normalize_symbols = [B_1, B_2, B_3, C_1, C_3, C_5]
normalize_equations = {}
new_symbols = {}
subs_normalize = []
for symbol in normalize_symbols:
    new_symbol = sp.Symbol('%sA' % symbol.name)
    new_symbols[symbol] = new_symbol
    eq = sp.Eq(new_symbol,symbol/A_44)

    normalize_equations[symbol]=eq

    subs_normalize.append((symbol, sp.solve(eq, symbol)[0]))

lhs = (roll_decay_equation_cubic.lhs/A_44).subs(subs_normalize).simplify()
roll_decay_equation_cubic_A = sp.Eq(lhs=lhs,rhs=0)


## Equivalt linearized damping:
B_e_equation = sp.Eq(B_e,B_1+8/(3*sp.pi)*omega0*phi_a*B_2)
B_e_equation_cubic = sp.Eq(B_e,B_1+8/(3*sp.pi)*omega0*phi_a*B_2 + 3/4*omega0**2*phi_a**2*B_3)


A_44_eq = sp.Eq(A_44, A44)
eqs = [
    A_44_eq,
    C_equation_linear,

]

omega0_eq = sp.Eq(omega0,sp.solve(eqs, omega0, GM)[1][0])
omega0_eq = omega0_eq.subs(C,C_1)

## Nondimensional damping Himeno:
lhs = B_44_hat
rhs = B_44/(rho*Disp*beam**2)*sp.sqrt(beam/(2*g))
B44_equation = sp.Eq(lhs, rhs)
omega0_equation_linear = omega0_equation.subs(C,sp.solve(C_equation_linear,C)[0])

omega_hat_equation = sp.Eq(omega_hat,omega*sp.sqrt(beam/(2*g)))
B44_hat_equation = sp.Eq(B_44_hat, B_44/(rho*Disp*beam**2)*sp.sqrt(beam/(2*g)))
B_1_hat_equation = sp.Eq(B_1_hat, B_1/(rho*Disp*beam**2)*sp.sqrt(beam/(2*g)))
B_e_hat_equation = sp.Eq(B_e_hat, B_e/(rho*Disp*beam**2)*sp.sqrt(beam/(2*g)))

B_2_hat_equation = sp.Eq(B_2_hat, B_2/(rho*Disp*beam**2)*sp.sqrt(beam/(2*g))**(0))

B44_hat_equation_quadratic = B44_hat_equation.subs(B_44,sp.solve(b44_quadratic_equation,B_44)[0])
omega0_hat_equation = omega_hat_equation.subs(omega,omega0)


## Analytical
diff_eq = sp.Eq(y.diff().diff() + 2 * zeta * omega0 * y.diff() + omega0 ** 2 * y, 0)
equation_D = sp.Eq(D, sp.sqrt(1 - zeta ** 2))

lhs = y
rhs = sp.exp(-zeta * omega0 * t) * (y0 * sp.cos(omega0 * D * t) + (y0_dot / (omega0 * D) + zeta * y0 / D) * sp.sin(omega0 * D * t))

analytical_solution_general = sp.Eq(lhs,rhs)

subs = [
    (y,phi),
    (y0, phi_0),
    (y0_dot, phi_0_dot),
    (y0_dotdot, phi_0_dotdot),
    (D,sp.solve(equation_D,D)[0]),
]

analytical_solution = analytical_solution_general.subs(subs)
analytical_phi1d = sp.Eq(phi_dot,sp.simplify(analytical_solution.rhs.diff(t)))
analytical_phi2d = sp.Eq(phi_dot_dot,sp.simplify(analytical_phi1d.rhs.diff(t)))

rhs = analytical_solution.rhs.args[1]*phi_0
lhs = phi_a
extinction_equation = sp.Eq(lhs,rhs)

xeta_equation = sp.Eq(zeta,
      sp.solve(extinction_equation,zeta)[0])

B_1_zeta_eq = sp.Eq(B_1, 2*zeta*omega0*A_44)
B_1_zeta_eq

### Simplified Ikeda
simplified_ikeda_equation = sp.Eq((B_F,
      B_W,
      B_E,
      B_BK,
       B_L,
      ),ikeda_simplified)


### Regression
regression_factor_equation = sp.Eq(B_e_hat,B_e_hat_0*B_e_factor)