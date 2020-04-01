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

    def run_after_init(self):
        signature = inspect.signature(self.estimator)
        self.parameter_names = list(signature.parameters.keys())[1:]

    @staticmethod
    def fit_derivation_omega(df, zeta, omega0):
        phi_old = df['phi']
        p_old = df['phi1d']

        phi2d = calculate_acceleration(omega0=omega0, phi1d=p_old, phi=phi_old, zeta=zeta)
        return phi2d

    @staticmethod
    def fit_integration_omega(df, zeta, omega0):

        phi_old = df['phi']
        p_old = df['phi1d']

        def roll_decay_time_step(states, t, zeta, omega0):
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
            zeta,
            d,
            omega0,
        )
        t = np.array(df.index)
        states = odeint(func=roll_decay_time_step, y0=states0, t=t, args=args)

        df = pd.DataFrame(index=t)

        phi = states[:, 0]
        return phi

    @property
    def estimator(self):

        fitter = self.get_fitter()

        def equation_no_omega(df, zeta):
            omega0 = float(df.iloc[0]['omega0'])
            return fitter(df=df, zeta=zeta, omega0=omega0)

        def equation_omega(df, zeta, omega0):
            return fitter(df=df, zeta=zeta, omega0=omega0)

        if self.omega_regression:
            return equation_omega
        else:
            return equation_no_omega