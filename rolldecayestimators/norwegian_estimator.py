import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from rolldecayestimators import DirectEstimator, measure as measure


class NorwegianEstimator(DirectEstimator):

    def __init__(self, maxfev = 4000, bounds=None, ftol=10**-10):
        super().__init__(maxfev=maxfev,bounds=bounds, ftol=ftol)
        self.phi_key = 'phi'  # Roll angle [rad]
        self.phi1d_key = 'phi1d'  # Roll velocity [rad/s]
        self.phi2d_key = 'phi2d'  # Roll acceleration [rad/s2]

    def fit(self, X, y=None):
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
        self.X = X.copy()
        self.is_fitted_ = True
        # `fit` should always return `self`

        self.calculate_amplitudes_and_damping()

        # Fitting part:
        self.linear_regression = LinearRegression()
        self.X_amplitudes['x'] = 2 / (3 * np.pi) * self.X_amplitudes['phi']

        self.linear_regression.fit(X=self.X_amplitudes[['x']], y=self.X_amplitudes['zeta_n'])

        omega0 = self.omega0
        self.parameters = {
            'd' : self.linear_regression.coef_[0],
            'zeta' : self.linear_regression.intercept_,
            'omega0' : omega0,
        }

        return self

