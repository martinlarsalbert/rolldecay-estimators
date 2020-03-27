"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sklearn.utils.validation import check_is_fitted
import inspect

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
    def _equation(df, zeta):

        phi_01d=0  # Assuming
        phi_0=df.iloc[0]['phi_0']
        omega0=df.iloc[0]['omega0']

        t = np.array(df.index - df.index[0])
        phi = analytical_solution_lambda(t=t,phi_0=phi_0, phi_01d=phi_01d, omega0=omega0, zeta=zeta)

        return phi

    @staticmethod
    def _equation_omega(df, omega0, zeta):
        phi_01d = 0  # Assuming
        phi_0 = df.iloc[0]['phi_0']

        t = np.array(df.index - df.index[0])
        phi = analytical_solution_lambda(t=t, phi_0=phi_0, phi_01d=phi_01d, omega0=omega0, zeta=zeta)

        return phi

    @property
    def equation(self):
        if self.omega_regression:
            return self._equation_omega
        else:
            return self._equation

    def simulate(self, t: np.ndarray, phi0: float, phi1d0:float, omega0: float, zeta: float,
                 **kwargs) -> pd.DataFrame:
        """
        Simulate a roll decay test using analytical solution
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param d: quadratic roll damping [-]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """

        df = pd.DataFrame(index=t)
        df['phi_0'] = phi0

        df['phi'] = self._equation_omega(df=df, omega0=omega0, zeta=zeta)


        t0=t-t[0]
        df['phi1d'] = analytical_solution_phi1d_lambda(t=t0,phi_0=phi0, phi_01d=phi1d0, omega0=omega0, zeta=zeta)
        df['phi2d'] = analytical_solution_phi2s_lambda(t=t0,phi_0=phi0, phi_01d=phi1d0, omega0=omega0, zeta=zeta)

        return df

    def fit(self, X, y=None, calculate_amplitudes_and_damping=True):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : Dummy not used.

        Returns
        -------
        self : object
            Returns self.
        """
        # X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        # `fit` should always return `self`

        self.X = X.copy()

        self.boundaries['zeta'] = (0,0.999)  # The equation produce division by zero for zeta=0
        self.X['phi_0'] = X.iloc[0]['phi']

        self.calculate_amplitudes_and_damping()
        self.X['omega0'] = self.omega0

        popt, pcov = curve_fit(f=self.equation, xdata=self.X, ydata=self.X['phi'], maxfev=self.maxfev,
                               ftol=self.ftol, bounds=self.get_bounds(),
                               p0=self.get_inital_guess())

        parameter_values = list(popt)
        parameters = dict(zip(self.parameter_names, parameter_values))

        self.parameters = parameters
        if not 'omega0' in self.parameters:
            self.parameters['omega0'] = self.omega0

        self.pcov = pcov

        return self
