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




class AnalyticalLinearEstimator(DirectEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    # Defining the diff equation for this estimator:
    rhs = -phi_dot_dot / (omega0 ** 2) - 2 * zeta / omega0 * phi_dot
    roll_diff_equation = sp.Eq(lhs=phi, rhs=rhs)
    acceleration = sp.Eq(lhs=phi, rhs=sp.solve(roll_diff_equation, phi.diff().diff())[0])
    functions = {
        'phi':lambdify(sp.solve(equations.analytical_solution, phi)[0]),
        'velocity':lambdify(sp.solve(equations.analytical_phi1d, phi_dot)[0]),
        'acceleration':lambdify(sp.solve(equations.analytical_phi2d, phi_dot_dot)[0]),
    }

    @property
    def parameter_names(self):
        signature = inspect.signature(self.calculate_acceleration)

        remove = ['phi_0','phi_01d', 't']
        if not self.omega_regression:
            remove.append('omega0')

        return list(set(signature.parameters.keys()) - set(remove))

    def estimator(self, x, xs):
        parameters = {key: x for key, x in zip(self.parameter_names, x)}

        if not self.omega_regression:
            parameters['omega0'] = self.omega0

        t = xs.index
        phi_0 = xs.iloc[0][self.phi_key]
        phi_01d = xs.iloc[0][self.phi1d_key]

        return self.functions['phi'](t=t,phi_0=phi_0,phi_01d=phi_01d,**parameters)

    def predict(self, X)->pd.DataFrame:

        check_is_fitted(self, 'is_fitted_')

        t = X.index
        phi_0 = X.iloc[0][self.phi_key]
        phi_01d = X.iloc[0][self.phi1d_key]

        df = pd.DataFrame(index=t)
        df['phi'] = self.functions['phi'](t=t, phi_0=phi_0, phi_01d=phi_01d, **self.parameters)
        df['phi1d'] = self.functions['velocity'](t=t, phi_0=phi_0, phi_01d=phi_01d, **self.parameters)
        df['phi2d'] = self.functions['acceleration'](t=t, phi_0=phi_0, phi_01d=phi_01d, **self.parameters)

        return df