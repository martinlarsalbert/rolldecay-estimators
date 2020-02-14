import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from rolldecayestimators import DirectEstimator, measure as measure


class NorwegianEstimator(DirectEstimator):

    def __init__(self):
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

        X_interpolated = measure.sample_increase(X=X)
        self.X_zerocrossings = measure.get_zerocrossings(X=X_interpolated)
        X_amplitudes = measure.calculate_amplitudes(X_zerocrossings=self.X_zerocrossings)
        self.X_amplitudes = self.calculate_damping(X_amplitudes=X_amplitudes)

        # Fitting part:

        self.linear_regression = LinearRegression()
        self.X_amplitudes['x'] = 2 / (3 * np.pi) * self.X_amplitudes['phi']

        self.linear_regression.fit(X=self.X_amplitudes[['x']], y=self.X_amplitudes['zeta_n'])

        T0=2*np.mean(self.X_amplitudes.index)
        omega0=2*np.pi/T0

        self.parameters = {
            'd' : self.linear_regression.coef_[0],
            'zeta' : self.linear_regression.intercept_,
            'omega0' : omega0,
        }

        return self

    @staticmethod
    def calculate_damping(X_amplitudes):

        df_decrements = pd.DataFrame()

        for i in range(len(X_amplitudes) - 1):
            s1 = X_amplitudes.iloc[i]
            s2 = X_amplitudes.iloc[i + 1]

            decrement = s1 / s2
            decrement.name = s1.name
            df_decrements = df_decrements.append(decrement)

        df_decrements['zeta_n'] = 1 / (2 * np.pi) * np.log(df_decrements['phi'])

        df_decrements['zeta_n']*=2  #!!! # Todo: Where did this one come from?

        X_amplitudes_new = X_amplitudes.copy()
        X_amplitudes_new = X_amplitudes_new.iloc[0:-1].copy()
        X_amplitudes_new['zeta_n'] = df_decrements['zeta_n'].copy()

        return X_amplitudes_new


    def plot_peaks(self, ax=None):
        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig,ax = plt.subplots()

        self.X.plot(y='phi', ax=ax)
        self.X_zerocrossings.plot(y='phi', ax=ax, style='r.')
        ax.plot([np.min(self.X.index),np.max(self.X.index)],[0,0],'m-')
        ax.set_title('Peaks')

    def plot_velocity(self, ax=None):
        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig,ax = plt.subplots()

        self.X.plot(y='phi1d', ax=ax)
        self.X_zerocrossings.plot(y='phi1d', ax=ax, style='r.')
        ax.plot([np.min(self.X.index), np.max(self.X.index)], [0, 0], 'm-')
        ax.set_title('Velocities')

    def plot_damping(self, ax=None):
        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig,ax = plt.subplots()

        df_amplitudes = self.X_amplitudes.copy()
        df_amplitudes['zeta_n_pred'] = self.linear_regression.predict(X=df_amplitudes[['x']])
        df_amplitudes.plot(x='x', y='zeta_n_pred', ax=ax, style='-')
        df_amplitudes.plot(x='x', y='zeta_n', ax=ax, style='.')
        ax.set_xlabel(r'$2/(3\pi)\Phi_n$', usetex=True)
        ax.set_ylabel(r'$\zeta_n$', usetex=True)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)