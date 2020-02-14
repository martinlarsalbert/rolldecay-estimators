"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import inspect
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from rolldecayestimators.equations_lambdify import calculate_acceleration_linear
from rolldecayestimators.simulation_linear import simulate

from rolldecayestimators.direct_estimator import DirectEstimator


class DirectLinearEstimator(DirectEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    @staticmethod
    def equation(df, omega0, zeta):
        phi_old = df['phi']
        p_old = df['phi1d']

        phi2d = calculate_acceleration_linear(omega0=omega0, p_old=p_old, phi_old=phi_old, zeta=zeta)
        return phi2d

    def do_simulation(self, t, phi0, phi1d0):
        return simulate(t=t, **self.parameters, phi0=phi0, phi1d0=phi1d0)