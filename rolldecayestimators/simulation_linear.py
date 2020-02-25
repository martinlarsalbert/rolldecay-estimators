from rolldecayestimators.direct_linear_estimator import calculate_acceleration
import numpy as np
import pandas as pd
from scipy.integrate import odeint

def roll_decay_time_step(states, t,omega0, zeta):
    # states:
    # [phi,phi1d]

    phi_old = states[0]
    p_old = states[1]

    phi1d = p_old
    phi2d = calculate_acceleration(omega0=omega0, phi1d=p_old, phi=phi_old, zeta=zeta)

    d_states_dt = np.array([phi1d, phi2d])

    return d_states_dt

def simulate(t :np.ndarray, phi0 :float, phi1d0 :float,omega0:float, zeta:float)->pd.DataFrame:
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
    states = odeint(func=roll_decay_time_step, y0=states0, t=t, args=args)

    df = pd.DataFrame(index=t)

    df['phi'] = states[:, 0]
    df['phi1d'] = states[:,1]

    return df
