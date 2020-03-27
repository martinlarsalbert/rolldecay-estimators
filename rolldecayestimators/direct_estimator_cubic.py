import numpy as np
import pandas as pd
from scipy.integrate import odeint
from rolldecayestimators import DirectEstimator
from rolldecayestimators.symbols import *
from rolldecayestimators import equations
from rolldecayestimators.substitute_dynamic_symbols import lambdify

## Cubic model:
b44_cubic_equation = sp.Eq(B_44,B_1*phi_dot + B_2*phi_dot*sp.Abs(phi_dot) + B_3*phi_dot**3 )
restoring_equation_cubic = sp.Eq(C_44,C_1*phi + C_3*phi**3 + C_5*phi**5)

subs = [
    (B_44,sp.solve(b44_cubic_equation,B_44)[0]),
    (C_44,sp.solve(restoring_equation_cubic,C_44)[0])
]
roll_decay_equation_cubic = equations.roll_decay_equation_general_himeno.subs(subs)
acceleration = sp.solve(equations.roll_decay_equation_cubic,phi_dot_dot)[0]
calculate_acceleration = lambdify(acceleration)

# Defining the diff equation for this estimator:

class DirectEstimatorCubic(DirectEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    @staticmethod
    def _equation(df, A_44, B_1, B_2, B_3, C_1, C_3, C_5):

        phi = df['phi']
        phi1d = df['phi1d']
        phi2d = calculate_acceleration(A_44, B_1, B_2, B_3, C_1, C_3, C_5, phi, phi1d)

        return phi2d

    @staticmethod
    def _omega_equation(df, A_44, B_1, B_2, B_3, C_1, C_3, C_5):
        raise ValueError('Not implemented')

    @staticmethod
    def roll_decay_time_step(states, t, A_44, B_1, B_2, B_3, C_1, C_3, C_5):
        # states:
        # [phi,phi1d]

        phi_old = states[0]
        p_old = states[1]

        phi1d = p_old
        phi2d = calculate_acceleration(A_44, B_1, B_2, B_3, C_1, C_3, C_5, phi_old, p_old)

        d_states_dt = np.array([phi1d, phi2d])

        return d_states_dt

    def simulate(self, t: np.ndarray, phi0: float, phi1d0: float, A_44, B_1, B_2, B_3, C_1, C_3, C_5) -> pd.DataFrame:
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
            A_44, B_1, B_2, B_3, C_1, C_3, C_5,
        )
        states = odeint(func=self.roll_decay_time_step, y0=states0, t=t, args=args)

        df = pd.DataFrame(index=t)

        df['phi'] = states[:, 0]
        df['phi1d'] = states[:, 1]

        return df

    def fit(self, X, y=None, calculate_amplitudes_and_damping=True):
        result = super().fit(X=X,y=y,calculate_amplitudes_and_damping=calculate_amplitudes_and_damping)
        if 'omega0' in result.parameters:
            result.parameters.pop('omega0')
        return result