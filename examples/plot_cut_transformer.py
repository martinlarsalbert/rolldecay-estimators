"""
=============================
Plotting Template Transformer
=============================

An example plot of :class:`rolldecayestimators.transformers.TemplateTransformer`
"""
import numpy as np
from matplotlib import pyplot as plt
from rolldecayestimators.transformers import CutTransformer

from rolldecayestimators.simulation import simulate

phi0 = np.deg2rad(2)
phi1d0 = 0

states0 = [phi0,phi1d0]
d = 0.076
T0 = 20
omega0 = 2*np.pi/T0
zeta = 0.044

N = 1000
t = np.linspace(0,120,N)

df = simulate(t=t, phi0=phi0, phi1d0=phi1d0, omega0=omega0, d=d, zeta=zeta)

fig,ax = plt.subplots()
df.plot(y='phi', ax=ax);

plt.show()

