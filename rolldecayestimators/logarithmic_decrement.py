import numpy as np
import sympy as sp
import pandas as pd

from rolldecayestimators import equations
from rolldecayestimators import symbols
from rolldecayestimators import lambdas

from rolldecayestimators.substitute_dynamic_symbols import lambdify

lambda_B_1_zeta = lambdify(sp.solve(equations.B_1_zeta_eq, symbols.B_1)[0])

eq_phi1d = sp.Eq(symbols.phi_dot_dot,
                 sp.solve(equations.roll_decay_equation_cubic_A, symbols.phi_dot_dot)[0])

accelaration_lambda = lambdify(sp.solve(eq_phi1d, symbols.phi_dot_dot)[0])


def find_peaks(df_state_space):
    df_state_space['phi_deg'] = np.rad2deg(df_state_space['phi'])

    mask = (np.sign(df_state_space['phi1d']) != np.sign(np.roll(df_state_space['phi1d'], -1))
            )
    mask[0] = False
    mask[-1] = False

    df_max = df_state_space.loc[mask].copy()
    df_max['id'] = np.arange(len(df_max)) + 1
    return df_max


def calculate_decrements(df_amplitudes):
    ## Calculate decrements:
    df_decrements = pd.DataFrame()
    for i in range(len(df_amplitudes) - 2):
        s1 = df_amplitudes.iloc[i]
        s2 = df_amplitudes.iloc[i + 2]
        decrement = s1 / s2
        decrement.name = s1.name
        df_decrements = df_decrements.append(decrement)

    return df_decrements


def calculate_zeta(df_decrements):
    zeta_n = 1 / (2 * np.pi) * np.log(df_decrements['phi'])
    return zeta_n

def calculate_B(zeta_n, A_44, omega0):

    ## Convert to B damping [Nm*s]
    B = lambda_B_1_zeta(A_44=A_44, omega0=omega0, zeta=zeta_n)
    B = np.array(B, dtype=float)
    return B

def estimate_amplitude(phi):
    phi = np.abs(phi)
    phi_a = (phi + np.roll(phi, -1) + np.roll(phi, -2)) / 3
    return phi_a