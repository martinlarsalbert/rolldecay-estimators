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

    @property
    def equation(self):
        return self._equation_omega

    @staticmethod
    def _equation(df, zeta):
        phi_old = df['phi']
        p_old = df['phi1d']
        omega0 = df.iloc[0]['omega0']

        phi2d = calculate_acceleration(omega0=omega0, phi1d=p_old, phi=phi_old, zeta=zeta)
        return phi2d

    #@staticmethod
    #def _equation_omega(df, omega0, zeta):
    #    phi_old = df['phi']
    #    p_old = df['phi1d']
    #
    #    phi2d = calculate_acceleration(omega0=omega0, phi1d=p_old, phi=phi_old, zeta=zeta)
    #    return phi2d

    @staticmethod
    def _equation_omega(df, omega0, zeta):
        phi_old = df['phi']
        p_old = df['phi1d']

        def roll_decay_time_step(states, t, omega0, zeta):
            # states:
            # [phi,phi1d]

            phi_old = states[0]
            p_old = states[1]

            phi1d = p_old
            phi2d = calculate_acceleration(omega0=omega0, phi1d=p_old, phi=phi_old, zeta=zeta)

            d_states_dt = np.array([phi1d, phi2d])

            return d_states_dt

        phi0 = df.iloc[0]['phi']
        phi1d0 = 0
        states0 = [phi0, phi1d0]
        args = (
            omega0,
            zeta,
        )
        t = np.array(df.index)
        states = odeint(func=roll_decay_time_step, y0=states0, t=t, args=args)

        df = pd.DataFrame(index=t)

        phi = states[:, 0]
        return phi

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
        #X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        # `fit` should always return `self`

        self.X = X.copy()
        if calculate_amplitudes_and_damping:
            self.calculate_amplitudes_and_damping()

            self.X['omega0'] = self.omega0


        popt, pcov = curve_fit(f=self.equation, xdata=self.X, ydata=self.X[self.phi_key],  maxfev=self.maxfev, ftol=self.ftol,
                               bounds=self.get_bounds(),
                               p0=self.get_inital_guess())

        parameter_values = list(popt)
        parameters = dict(zip(self.parameter_names, parameter_values))

        self.parameters=parameters

        if not 'omega0' in self.parameters:
            self.parameters['omega0'] = self.omega0

        self.pcov = pcov

        return self


    def roll_decay_time_step(self, states, t, omega0, zeta):
        # states:
        # [phi,phi1d]

        phi_old = states[0]
        p_old = states[1]

        phi1d = p_old
        phi2d = calculate_acceleration(omega0=omega0, phi1d=p_old, phi=phi_old, zeta=zeta)

        d_states_dt = np.array([phi1d, phi2d])

        return d_states_dt

    def simulate(self, t: np.ndarray, phi0: float, phi1d0: float, omega0: float, zeta: float) -> pd.DataFrame:
        """
        Simulate a roll decay test using the quadratic method.
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """

        states0 = [phi0, phi1d0]
        args = (
            omega0,
            zeta,
        )
        states = odeint(func=self.roll_decay_time_step, y0=states0, t=t, args=args)

        df = pd.DataFrame(index=t)

        df['phi'] = states[:, 0]
        df['phi1d'] = states[:, 1]
        df['phi2d'] = calculate_acceleration(omega0=omega0, phi1d=df['phi1d'], phi=df['phi'], zeta=zeta)

        return df