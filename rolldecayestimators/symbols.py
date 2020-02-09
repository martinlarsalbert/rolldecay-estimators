import rolldecayestimators.special_symbol as ss
import sympy as sp
import sympy.physics.mechanics as me

t = ss.Symbol(name='t',description='time',unit='s')

phi = me.dynamicsymbols('phi')  # Roll angle
phi_dot = phi.diff()
phi_dot_dot = phi_dot.diff()

zeta = sp.Symbol('zeta') # Linear roll damping coefficeint
omega0 = sp.Symbol('omega0')  # Natural roll frequency
d = sp.Symbol('d')  # Nonlinear roll damping coefficient