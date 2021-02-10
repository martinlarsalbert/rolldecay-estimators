import numpy as np
import pandas as pd
from scipy.integrate import odeint
from rolldecayestimators import DirectEstimator
from rolldecayestimators.symbols import *
from rolldecayestimators import equations, symbols
from rolldecayestimators.substitute_dynamic_symbols import lambdify, run
from sklearn.utils.validation import check_is_fitted
from rolldecayestimators.estimator import RollDecay



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

    eqs = [equations.C_equation_linear,
           equations.omega0_equation,
           A44_equation,
           ]
    omgea0_equation = sp.Eq(symbols.omega0, sp.solve(eqs, symbols.A_44, symbols.C, symbols.omega0)[0][2])
    functions['omega0'] = lambdify(sp.solve(omgea0_equation,symbols.omega0)[0])

    def __init__(self, maxfev=1000, bounds={}, ftol=10 ** -15, p0={}, fit_method='integration'):

        new_bounds={
            'B_1A':(0, np.inf),  # Assuming only positive coefficients
        #    'B_2A': (0, np.inf),  # Assuming only positive coefficients
        #    'B_3A': (0, np.inf),  # Assuming only positive coefficients
        }
        new_bounds.update(bounds)
        bounds=new_bounds

        super().__init__(maxfev=maxfev, bounds=bounds, ftol=ftol, p0=p0, fit_method=fit_method, omega_regression=True)


    @classmethod
    def load(cls, B_1A:float, B_2A:float, B_3A:float, C_1A:float, C_3A:float, C_5A:float, X=None, **kwargs):
        """
        Load data and parameters from an existing fitted estimator

        A_44 is total roll intertia [kg*m**2] (including added mass)

        Parameters
        ----------
        B_1A
            B_1/A_44 : linear damping
        B_2A
            B_2/A_44 : quadratic damping
        B_3A
            B_3/A_44 : cubic damping
        C_1A
            C_1/A_44 : linear stiffness
        C_3A
            C_3/A_44 : cubic stiffness
        C_5A
            C_5/A_44 : pentatonic stiffness

        X : pd.DataFrame
            DataFrame containing the measurement that this estimator fits (optional).
        Returns
        -------
        estimator
            Loaded with parameters from data and maybe also a loaded measurement X
        """
        data={
            'B_1A':B_1A,
            'B_2A':B_2A,
            'B_3A':B_3A,
            'C_1A':C_1A,
            'C_3A':C_3A,
            'C_5A':C_5A,
        }

        return super(cls, cls)._load(data=data, X=X)


    def calculate_additional_parameters(self, A44):
        check_is_fitted(self, 'is_fitted_')

        parameters_additional = {}

        for key, value in self.parameters.items():
            symbol_key = sp.Symbol(key)
            new_key = key[0:-1]
            symbol_new_key = ss.Symbol(new_key)

            if symbol_new_key in equations.normalize_equations:
                normalize_equation = equations.normalize_equations[symbol_new_key]
                solution = sp.solve(normalize_equation,symbol_new_key)[0]
                new_value = solution.subs([(symbol_key,value),
                                           (symbols.A_44,A44),

                               ])

                parameters_additional[new_key]=new_value

        return parameters_additional


    def result_for_database(self, meta_data={}):
        s = super().result_for_database(meta_data=meta_data)

        inputs=pd.Series(meta_data)

        inputs['m'] = inputs['Volume']*inputs['rho']
        parameters = pd.Series(self.parameters)
        inputs = parameters.combine_first(inputs)

        s['A_44'] = run(self.functions['A44'], inputs=inputs)
        parameters_additional = self.calculate_additional_parameters(A44=s['A_44'])
        s.update(parameters_additional)

        inputs['A_44'] = s['A_44']
        s['omega0'] = run(function=self.functions['omega0'], inputs=inputs)

        self.results = s  # Store it also

        return s


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

    @classmethod
    def load(cls, B_1A:float, B_2A:float, C_1A:float, X=None, **kwargs):
        """
        Load data and parameters from an existing fitted estimator

        A_44 is total roll intertia [kg*m**2] (including added mass)

        Parameters
        ----------
        B_1A
            B_1/A_44 : linear damping
        B_2A
            B_2/A_44 : quadratic damping
        C_1A
            C_1/A_44 : linear stiffness

        X : pd.DataFrame
            DataFrame containing the measurement that this estimator fits (optional).
        Returns
        -------
        estimator
            Loaded with parameters from data and maybe also a loaded measurement X
        """
        data={
            'B_1A':B_1A,
            'B_2A':B_2A,
            'C_1A':C_1A,
        }

        return super(cls, cls)._load(data=data, X=X)


class EstimatorQuadraticBandC(EstimatorCubic):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    ## Quadratic model:
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

class EstimatorLinear(EstimatorCubic):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    ## Linear model:
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

    @classmethod
    def load(cls, B_1A:float, C_1A:float, X=None, **kwargs):
        """
        Load data and parameters from an existing fitted estimator

        A_44 is total roll intertia [kg*m**2] (including added mass)

        Parameters
        ----------
        B_1A
            B_1/A_44 : linear damping
        C_1A
            C_1/A_44 : linear stiffness

        X : pd.DataFrame
            DataFrame containing the measurement that this estimator fits (optional).
        Returns
        -------
        estimator
            Loaded with parameters from data and maybe also a loaded measurement X
        """
        data={
            'B_1A':B_1A,
            'C_1A':C_1A,
        }

        return super(cls, cls)._load(data=data, X=X)


