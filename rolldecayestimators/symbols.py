import rolldecayestimators.special_symbol as ss
import sympy as sp
import sympy.physics.mechanics as me

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
dGM = ss.Symbol(name='dGM', description='metacentric height correction', unit='m/rad')
omega = ss.Symbol(name='omega', description='Frequency of external moment', unit='Nm')
L_pp = ss.Symbol(name='L_pp',description='ship perpendicular length',unit='m')
beam = ss.Symbol(name='beam',description='ship beam',unit='m')

phi = me.dynamicsymbols('phi')  # Roll angle
#phi = ss.Symbol(name='phi', description='Roll angle', unit='rad')  # Roll angle
phi_dot = phi.diff()
phi_dot_dot = phi_dot.diff()
phi_a = ss.Symbol(name='phi_a', description='Initial roll amplitude', unit='rad')

zeta = sp.Symbol('zeta') # Linear roll damping coefficeint
omega0 = sp.Symbol('omega0')  # Natural roll frequency
d = sp.Symbol('d')  # Nonlinear roll damping coefficient

A_44 = ss.Symbol(name='A_44', description='General roll inertia', unit='kg*m**2')

B_1,B_2,B_3 = sp.symbols('B_1 B_2 B_3')
C = sp.Symbol(name='C')  # Introducing a helper coefficient C
C_1, C_3, C_5 = sp.symbols('C_1 C_3 C_5')

B_e = ss.Symbol(name='B_e', description='Equivalen linearized damping', unit='Nm/(rad/s)')
B_44_hat = ss.Symbol(name='B_44_hat', description='Nondimensional damping', unit='-')

## Functions:
GZ = sp.Function('GZ')(phi)
B_44 = sp.Function('B_{44}')(phi_dot)
C_44 = sp.Function('C_{44}')(phi)
M_44 = sp.Function('M_{44}')(omega*t)

## Analytical
delta = sp.symbols('delta')
y = me.dynamicsymbols('y')
y0 = me.dynamicsymbols('y0')
y0_dot = y0.diff()
y0_dotdot = y0_dot.diff()
D = sp.symbols('D')

phi_0 = me.dynamicsymbols('phi_0')
phi_0_dot = phi_0.diff()
phi_0_dotdot = phi_0_dot.diff()



