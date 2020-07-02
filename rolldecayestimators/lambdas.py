import sympy as sp
from rolldecayestimators.substitute_dynamic_symbols import lambdify
from rolldecayestimators import equations
from rolldecayestimators import symbols

B44_lambda = lambdify(sp.solve(equations.B44_hat_equation_quadratic,symbols.B_44_hat)[0])
B_1_hat_lambda = lambdify(sp.solve(equations.B_1_hat_equation, symbols.B_1_hat)[0])
B_e_hat_lambda = lambdify(sp.solve(equations.B_e_hat_equation, symbols.B_e_hat)[0])
B_2_hat_lambda = lambdify(sp.solve(equations.B_2_hat_equation, symbols.B_2_hat)[0])

omega0_lambda = lambdify(sp.solve(equations.omega0_hat_equation,symbols.omega_hat)[0])
B_e_lambda = lambdify(sp.solve(equations.B_e_equation, symbols.B_e)[0])
B_e_lambda_cubic = lambdify(sp.solve(equations.B_e_equation_cubic, symbols.B_e)[0])

