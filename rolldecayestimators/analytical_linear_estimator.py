"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from rolldecayestimators.substitute_dynamic_symbols import lambdify
from rolldecayestimators.symbols import *
from rolldecayestimators import equations

from rolldecayestimators.direct_estimator import DirectEstimator

analytical_solution_lambda = lambdify(sp.solve(equations.analytical_solution,phi)[0])
analytical_solution_phi1d_lambda = lambdify(sp.solve(equations.analytical_phi1d,phi_dot)[0])
analytical_solution_phi2s_lambda = lambdify(sp.solve(equations.analytical_phi2d,phi_dot_dot)[0])


class AnalyticalLinearEstimator(DirectEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    @staticmethod
    def equation(df,phi_0, omega0, zeta):

        phi_01d=0  # Assuming zero here!

        phi = analytical_solution_lambda(t=df.index,phi_0=phi_0, phi_01d=phi_01d, omega0=omega0, zeta=zeta)

        return phi


    def simulate(self, t: np.ndarray, phi_0, omega0: float, zeta: float,**kwargs) -> pd.DataFrame:
        """
        Simulate a roll decay test using analytical solution
        :param t: time vector to be simulated [s]
        :param phi_0: initial roll angle [rad]
        :param omega0: roll natural frequency[rad/s]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi', 'phi1d', 'phi2d'
        """


        df = pd.DataFrame(index=t)
        df['phi'] = self.equation(df=df, phi_0=phi_0, omega0=omega0, zeta=zeta)
        phi_01d=0
        df['phi1d'] = analytical_solution_phi1d_lambda(t=df.index,phi_0=phi_0, phi_01d=phi_01d, omega0=omega0, zeta=zeta)
        df['phi2d'] = analytical_solution_phi2s_lambda(t=df.index,phi_0=phi_0, phi_01d=phi_01d, omega0=omega0, zeta=zeta)

        return df