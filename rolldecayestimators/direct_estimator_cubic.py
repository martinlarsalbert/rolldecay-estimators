import numpy as np
import pandas as pd
from scipy.integrate import odeint
from rolldecayestimators import DirectEstimator
from rolldecayestimators.symbols import *
from rolldecayestimators import equations, symbols
from rolldecayestimators.substitute_dynamic_symbols import lambdify




class EstimatorCubic(DirectEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    ## Cubic model:
    b44_cubic_equation = sp.Eq(B_44, B_1 * phi_dot + B_2 * phi_dot * sp.Abs(phi_dot) + B_3 * phi_dot ** 3)
    restoring_equation_cubic = sp.Eq(C_44, C_1 * phi + C_3 * phi ** 3 + C_5 * phi ** 5)

    subs = [
        (B_44, sp.solve(b44_cubic_equation, B_44)[0]),
        (C_44, sp.solve(restoring_equation_cubic, C_44)[0])
    ]
    roll_decay_equation = equations.roll_decay_equation_general_himeno.subs(subs)
    # Normalizing with A_44:
    lhs = (roll_decay_equation.lhs / A_44).subs(equations.subs_normalize).simplify()
    roll_decay_equation_A = sp.Eq(lhs=lhs, rhs=0)

    acceleration = sp.solve(roll_decay_equation_A, phi_dot_dot)[0]
    functions = {
                'acceleration':lambdify(acceleration)
                }

    C_1_equation = equations.C_equation_linear.subs(symbols.C, symbols.C_1)  # C_1 = GM*gm

    eqs = [
        C_1_equation,
        equations.normalize_equations[symbols.C_1]
    ]

    A44_equation = sp.Eq(symbols.A_44, sp.solve(eqs, symbols.C_1, symbols.A_44)[symbols.A_44])
    functions['A44'] = lambdify(sp.solve(A44_equation, symbols.A_44)[0])

    def __init__(self, maxfev=4000, bounds={}, ftol=10 ** -15, p0={}, fit_method='derivation'):
        super().__init__(maxfev=maxfev, bounds=bounds, ftol=ftol, p0=p0, fit_method=fit_method, omega_regression=True)

    def simulate(self, t :np.ndarray, phi0 :float, phi1d0 :float, B_1A, B_2A, B_3A, C_1A, C_3A, C_5A,)->pd.DataFrame:
        """
        Simulate a roll decay test using the quadratic method.
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """
        parameters={
            'B_1A':B_1A,
            'B_2A':B_2A,
            'B_3A':B_3A,
            'C_1A':C_1A,
            'C_3A':C_3A,
            'C_5A':C_5A,
        }
        return self._simulate(t=t, phi0=phi0, phi1d0=phi1d0, parameters=parameters)

class EstimatorQuadraticB(EstimatorCubic):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    ## Cubic model:
    b44_quadratic_equation = sp.Eq(B_44, B_1 * phi_dot + B_2 * phi_dot * sp.Abs(phi_dot))
    restoring_equation_quadratic = sp.Eq(C_44, C_1 * phi)

    subs = [
        (B_44, sp.solve(b44_quadratic_equation, B_44)[0]),
        (C_44, sp.solve(restoring_equation_quadratic, C_44)[0])
    ]
    roll_decay_equation = equations.roll_decay_equation_general_himeno.subs(subs)
    # Normalizing with A_44:
    lhs = (roll_decay_equation.lhs / A_44).subs(equations.subs_normalize).simplify()
    roll_decay_equation_A = sp.Eq(lhs=lhs, rhs=0)

    acceleration = sp.solve(roll_decay_equation_A, phi_dot_dot)[0]
    functions = dict(EstimatorCubic.functions)
    functions['acceleration'] = lambdify(acceleration)


    def simulate(self, t :np.ndarray, phi0 :float, phi1d0 :float, B_1, B_2, C_1)->pd.DataFrame:
        """
        Simulate a roll decay test using the quadratic method.
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """
        parameters={
            'B_1':B_1,
            'B_2':B_2,
            'C_1':C_1,
        }
        return self._simulate(t=t, phi0=phi0, phi1d0=phi1d0, parameters=parameters)

class EstimatorQuadraticBandC(EstimatorCubic):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    ## Cubic model:
    b44_quadratic_equation = sp.Eq(B_44, B_1 * phi_dot + B_2 * phi_dot * sp.Abs(phi_dot))
    restoring_equation_quadratic = sp.Eq(C_44, C_1 * phi + C_3 * phi ** 3)

    subs = [
        (B_44, sp.solve(b44_quadratic_equation, B_44)[0]),
        (C_44, sp.solve(restoring_equation_quadratic, C_44)[0])
    ]
    roll_decay_equation = equations.roll_decay_equation_general_himeno.subs(subs)
    # Normalizing with A_44:
    lhs = (roll_decay_equation.lhs / A_44).subs(equations.subs_normalize).simplify()
    roll_decay_equation_A = sp.Eq(lhs=lhs, rhs=0)

    acceleration = sp.solve(roll_decay_equation_A, phi_dot_dot)[0]
    functions = dict(EstimatorCubic.functions)
    functions['acceleration'] = lambdify(acceleration)


    def simulate(self, t :np.ndarray, phi0 :float, phi1d0 :float, B_1, B_2, C_1, C_3)->pd.DataFrame:
        """
        Simulate a roll decay test using the quadratic method.
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """
        parameters={
            'B_1':B_1,
            'B_2':B_2,
            'C_1':C_1,
            'C_3':C_3,
        }
        return self._simulate(t=t, phi0=phi0, phi1d0=phi1d0, parameters=parameters)

class EstimatorLinear(EstimatorCubic):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    ## Cubic model:
    b44_linear_equation = sp.Eq(B_44, B_1 * phi_dot)
    restoring_linear_quadratic = sp.Eq(C_44, C_1 * phi)


    subs = [
        (B_44, sp.solve(b44_linear_equation, B_44)[0]),
        (C_44, sp.solve(restoring_linear_quadratic, C_44)[0])
    ]
    roll_decay_equation = equations.roll_decay_equation_general_himeno.subs(subs)
    # Normalizing with A_44:
    lhs = (roll_decay_equation.lhs / A_44).subs(equations.subs_normalize).simplify()
    roll_decay_equation_A = sp.Eq(lhs=lhs, rhs=0)

    acceleration = sp.solve(roll_decay_equation_A, phi_dot_dot)[0]
    functions = dict(EstimatorCubic.functions)
    functions['acceleration'] = lambdify(acceleration)

    def simulate(self, t :np.ndarray, phi0 :float, phi1d0 :float, B_1, C_1)->pd.DataFrame:
        """
        Simulate a roll decay test using the quadratic method.
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """
        parameters={
            'B_1':B_1,
            'C_1':C_1,
        }
        return self._simulate(t=t, phi0=phi0, phi1d0=phi1d0, parameters=parameters)

