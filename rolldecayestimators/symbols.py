import rolldecayestimators.special_symbol as ss
import sympy as sp
import sympy.physics.mechanics as me

t = ss.Symbol(name='t',description='time',unit='s')
I = ss.Symbol(name='I',description='total roll inertia',unit='kg*m**2')
B = ss.Symbol(name='B',description='total roll damping',unit='kg*m*/s')
rho = ss.Symbol(name='rho',description='water density',unit='kg/m3')
g = ss.Symbol(name='g',description='acceleration of gravity',unit='m/s**2')
Disp = ss.Symbol(name='Disp',description='displacement',unit='m**3')
M_x = ss.Symbol(name='M_x',description='External roll moment',unit='Nm')
m = ss.Symbol(name='m',description='mass of ship',unit='kg')
GM = ss.Symbol(name='GM', description='metacentric height', unit='m')

phi = me.dynamicsymbols('phi')  # Roll angle
phi_dot = phi.diff()
phi_dot_dot = phi_dot.diff()

zeta = sp.Symbol('zeta') # Linear roll damping coefficeint
omega0 = sp.Symbol('omega0')  # Natural roll frequency
d = sp.Symbol('d')  # Nonlinear roll damping coefficient

## Functions:
GZ = sp.Function('GZ')(phi)
