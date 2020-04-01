import inspect
from scipy.optimize import least_squares
from scipy.integrate import odeint
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd

from rolldecayestimators.substitute_dynamic_symbols import lambdify
from rolldecayestimators.symbols import *


class RollDecay(BaseEstimator):

    # Defining the diff equation for this estimator:
    rhs = -phi_dot_dot / (omega0 ** 2) - 2 * zeta / omega0 * phi_dot - d * sp.Abs(phi_dot) * phi_dot / (omega0 ** 2)
    roll_diff_equation = sp.Eq(lhs=phi, rhs=rhs)
    acceleration = sp.Eq(lhs=phi, rhs=sp.solve(roll_diff_equation, phi.diff().diff())[0])
    functions = (lambdify(acceleration.rhs),)

    def __init__(self, maxfev = 4000, bounds={}, ftol=10**-10, p0={}, omega_regression=False,fit_method='derivation'):
        self.is_fitted_ = False

        self.phi_key = 'phi'  # Roll angle [rad]
        self.phi1d_key = 'phi1d'  # Roll velocity [rad/s]
        self.phi2d_key = 'phi2d'  # Roll acceleration [rad/s2]

    @property
    def calculate_acceleration(self):
        return self.functions[0]

    @property
    def parameter_names(self):
        signature = inspect.signature(self.calculate_acceleration)
        return list(set(signature.parameters.keys()) - set([self.phi_key, self.phi1d_key]))

    @staticmethod
    def error(x, self, xs, ys):
        return ys - self.equation(x, xs)

    def equation(self, x, xs):
        parameters = {key: x for key, x in zip(self.parameter_names, x)}

        phi = xs[self.phi_key]
        phi1d = xs[self.phi1d_key]

        acceleration = self.calculate_acceleration(phi=phi, phi1d=phi1d, **parameters)
        return acceleration

    def fit(self, X):
        kwargs = {'self': self,
                  'xs': X,
                  'ys': X[self.phi2d_key]}

        self.result = least_squares(fun=self.error, x0=[0.5, 0.5, 0.5], kwargs=kwargs)
        self.parameters = {key: x for key, x in zip(self.parameter_names, self.result.x)}

        self.is_fitted_ = True

    @staticmethod
    def roll_decay_time_step(states, t, self):
        # states:
        # [phi,phi1d]

        phi_old = states[0]
        p_old = states[1]

        phi1d = p_old
        calculate_acceleration = self.calculate_acceleration
        phi2d = calculate_acceleration(phi1d=p_old, phi=phi_old, **self.parameters)

        d_states_dt = np.array([phi1d, phi2d])

        return d_states_dt

    def predict(self, X)->pd.DataFrame:

        check_is_fitted(self, 'is_fitted_')

        phi0 = X[self.phi_key].iloc[0]
        phi1d0 = X[self.phi1d_key].iloc[0]
        states0 = [phi0, phi1d0]

        t = np.array(X.index)
        states = odeint(self.roll_decay_time_step, y0=states0, t=t, args=(self,))
        df = pd.DataFrame(index=t)

        df[self.phi_key] = states[:, 0]
        df[self.phi1d_key] = states[:, 1]

        return df

    def score(self, X=None, y=None, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
        ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
        A constant model that always predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic
            objects instead, shape = (n_samples, n_samples_fitted), where n_samples_fitted is the number of samples
            used in the fitting for the estimator.

        y : Dummy not used

        sample_weight : Dummy

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.

        """

        y_true, y_pred = self.true_and_prediction(X=X)

        return r2_score(y_true=y_true, y_pred=y_pred)

    def true_and_prediction(self, X=None):

        if X is None:
            X=self.X

        y_true = X[self.phi_key]
        df_sim = self.predict(X)
        y_pred = df_sim[self.phi_key]
        return y_true, y_pred