import numpy as np
import pandas as pd
from scipy.integrate import odeint
from rolldecayestimators import DirectEstimator
from rolldecayestimators.symbols import *
from rolldecayestimators.substitute_dynamic_symbols import lambdify

dGM = sp.symbols('dGM')
lhs = phi_dot_dot + 2*zeta*omega0*phi_dot + omega0**2*phi+(dGM*phi*sp.Abs(phi)) + d*sp.Abs(phi_dot)*phi_dot
roll_diff_equation = sp.Eq(lhs=lhs, rhs=0)
acceleration = sp.Eq(lhs=phi, rhs=sp.solve(roll_diff_equation, phi.diff().diff())[0])
calculate_acceleration = lambdify(acceleration.rhs)

# Defining the diff equation for this estimator:

class DirectEstimatorImproved(DirectEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    @staticmethod
    def estimator(df, omega0, zeta, dGM, d):
        phi = df['phi']
        phi1d = df['phi1d']

        phi2d = calculate_acceleration(omega0=omega0, phi=phi, phi1d=phi1d, zeta=zeta, dGM=dGM, d=d)

        return phi2d

    @staticmethod
    def roll_decay_time_step(states, t, omega0, zeta, dGM, d):
        # states:
        # [phi,phi1d]

        phi_old = states[0]
        p_old = states[1]

        phi1d = p_old
        phi2d = calculate_acceleration(omega0=omega0, phi=phi_old, phi1d=p_old, zeta=zeta, dGM=dGM, d=d)

        d_states_dt = np.array([phi1d, phi2d])

        return d_states_dt

    def simulate(self, t: np.ndarray, phi0: float, phi1d0: float, omega0: float, zeta: float, dGM: float,
                 d: float) -> pd.DataFrame:
        """
        Simulate a roll decay test using the quadratic method.
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param d: quadratic roll damping [-]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """

        states0 = [phi0, phi1d0]
        args = (
            omega0,
            zeta,
            dGM,
            d,
        )
        states = odeint(func=self.roll_decay_time_step, y0=states0, t=t, args=args)

        df = pd.DataFrame(index=t)

        df['phi'] = states[:, 0]
        df['phi1d'] = states[:, 1]

        return df
