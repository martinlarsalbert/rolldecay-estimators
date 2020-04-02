import numpy as np
import pandas as pd
from scipy.integrate import odeint
from rolldecayestimators import DirectEstimator
from rolldecayestimators.symbols import *
from rolldecayestimators import equations
from rolldecayestimators.substitute_dynamic_symbols import lambdify




class DirectEstimatorCubic(DirectEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    ## Cubic model:
    b44_cubic_equation = sp.Eq(B_44, B_1 * phi_dot + B_2 * phi_dot * sp.Abs(phi_dot) + B_3 * phi_dot ** 3)
    restoring_equation_cubic = sp.Eq(C_44, C_1 * phi + C_3 * phi ** 3 + C_5 * phi ** 5)

    subs = [
        (B_44, sp.solve(b44_cubic_equation, B_44)[0]),
        (C_44, sp.solve(restoring_equation_cubic, C_44)[0])
    ]
    roll_decay_equation_cubic = equations.roll_decay_equation_general_himeno.subs(subs)
    acceleration = sp.solve(equations.roll_decay_equation_cubic, phi_dot_dot)[0]
    functions = (lambdify(acceleration),)

    def __init__(self, maxfev=4000, bounds={}, ftol=10 ** -20, p0={}, fit_method='derivation'):
        super().__init__(maxfev=maxfev, bounds=bounds, ftol=ftol, p0=p0, fit_method=fit_method, omega_regression=True)

    def simulate(self, t :np.ndarray, phi0 :float, phi1d0 :float, A_44, B_1, B_2, B_3, C_1, C_3, C_5,)->pd.DataFrame:
        """
        Simulate a roll decay test using the quadratic method.
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """
        parameters={
            'A_44':A_44,
            'B_1':B_1,
            'B_2':B_2,
            'B_3':B_3,
            'C_1':C_1,
            'C_3':C_3,
            'C_5':C_5,
        }
        return self._simulate(t=t, phi0=phi0, phi1d0=phi1d0, parameters=parameters)

