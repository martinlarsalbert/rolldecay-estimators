from rolldecayestimators import equations
from rolldecayestimators.substitute_dynamic_symbols import lambdify

calculate_acceleration = lambdify(equations.acceleration_equation.rhs)
calculate_velocity = lambdify(equations.velocity_equation.rhs)