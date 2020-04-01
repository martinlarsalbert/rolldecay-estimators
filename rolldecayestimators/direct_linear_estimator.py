"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from rolldecayestimators.substitute_dynamic_symbols import lambdify
from rolldecayestimators.symbols import *
import inspect
from scipy.optimize import curve_fit

from rolldecayestimators.direct_estimator import DirectEstimator


lhs = phi_dot_dot + 2*zeta*omega0*phi_dot + omega0**2*phi
roll_diff_equation = sp.Eq(lhs=lhs,rhs=0)
acceleration = sp.Eq(lhs=phi, rhs=sp.solve(roll_diff_equation, phi.diff().diff())[0])
calculate_acceleration = lambdify(acceleration.rhs)


class DirectLinearEstimator(DirectEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    # Defining the diff equation for this estimator:
    rhs = -phi_dot_dot/(omega0**2) - 2*zeta/omega0*phi_dot
    roll_diff_equation = sp.Eq(lhs=phi, rhs=rhs)
    acceleration = sp.Eq(lhs=phi, rhs=sp.solve(roll_diff_equation, phi.diff().diff())[0])
    functions = (lambdify(acceleration.rhs),)