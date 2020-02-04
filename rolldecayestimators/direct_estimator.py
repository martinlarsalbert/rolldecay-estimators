"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import inspect
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class DirectEstimator(BaseEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, calculate_acceleration,simulate,p0=[0.01,0.22,0.01]):
        self.calculate_acceleration = calculate_acceleration
        self.simulate = simulate
        self.p0 = p0
        self.phi_key = 'phi'  # Roll angle [rad]
        self.phi1d_key = 'phi1d'  # Roll velocity [rad/s]
        self.phi2d_key = 'phi2d'  # Roll acceleration [rad/s2]
        self.maxfev = 4000
        self.bounds = (0,np.inf,)

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
        self.is_fitted_ = True
        # `fit` should always return `self`

        def f_direct(df, d, omega0, zeta):
            phi_old = df[self.phi_key]
            p_old = df[self.phi1d_key]

            phi2d = self.calculate_acceleration(d=d, omega0=omega0, p_old=p_old, phi_old=phi_old, zeta=zeta)
            return phi2d

        popt, pcov = curve_fit(f=f_direct, xdata=X, ydata=X[self.phi2d_key], p0=self.p0, maxfev=self.maxfev,
                               bounds=self.bounds)
        self.X = X.copy()

        signature = inspect.signature(f_direct)
        parameter_names = list(signature.parameters.keys())[1:]

        parameter_values = list(popt)
        parameters = dict(zip(parameter_names, parameter_values))

        self.parameters =parameters
        self.pcov = pcov

        return self


    def score(self, X, y=None, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
        ((y_true - y_pred) ** 2).mean() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).mean().
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
        y_true = X[self.phi_key]
        df_sim = self.predict(X)
        y_pred = df_sim[self.phi_key]
        u = ((y_true - y_pred) ** 2).mean()
        v = ((y_true - y_true.mean()) ** 2).mean()
        return (1 - u/v)


    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        #X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        phi0 = X[self.phi_key].iloc[0]
        phi1d0 = X[self.phi1d_key].iloc[0]

        df_sim = self.simulate(t=X.index, **self.parameters, phi0=phi0, phi1d0=phi1d0)

        #return np.ones(X.shape[0], dtype=np.int64)
        return df_sim

    def calculate_average_linear_damping(self,phi_a=None):
        """
        Calculate the average linear damping
        In line with Himeno this equivalent linear damping is calculated as
        Parameters
        ----------
        phi_a : float, default = None
            linearize around this average angle [rad]
            phi_a is calculated based on data if None

        Returns
        -------
        average_linear_damping

        """
        check_is_fitted(self, 'is_fitted_')

        zeta = self.parameters['zeta']
        d = self.parameters['d']

        if phi_a is None:
            phi_a = self.X[self.phi_key].abs().mean()

        return zeta + 4/(3*np.pi)*d*phi_a

class NorwegianEstimator(DirectEstimator):

    def __init__(self, calculate_acceleration,simulate):
        self.calculate_acceleration = calculate_acceleration
        self.simulate = simulate
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

        X_interpolated = self.sample_increase(X=X)
        self.X_zerocrossings = self.get_zerocrossings(X=X_interpolated)
        X_amplitudes = self.calculate_amplitudes(X_zerocrossings=self.X_zerocrossings)
        self.X_amplitudes = self.calculate_damping(X_amplitudes=X_amplitudes)

        # Fitting part:

        self.linear_regression = LinearRegression()
        self.X_amplitudes['x'] = 2 / (3 * np.pi) * self.X_amplitudes['phi']

        self.linear_regression.fit(X=self.X_amplitudes[['x']], y=self.X_amplitudes['zeta_n'])

        T0=2*np.mean(np.diff(self.X_amplitudes.index))
        omega0=2*np.pi/T0

        self.parameters = {
            'd' : self.linear_regression.coef_[0],
            'zeta' : self.linear_regression.intercept_,
            'omega0' : omega0,
        }

        return self

    @staticmethod
    def sample_increase(X):
        N = len(X) * 10
        t_interpolated = np.linspace(X.index[0], X.index[-1], N)
        X_interpolated = pd.DataFrame(index=t_interpolated)

        for key, values in X.items():
            X_interpolated[key] = np.interp(t_interpolated, values.index, values)

        return X_interpolated

    @staticmethod
    def get_zerocrossings(X):

        phi1d = np.array(X['phi1d'])
        index = np.arange(0, len(X.index))
        index_later = np.roll(index, shift=-1)
        mask = (
                ((phi1d[index] > 0) &
                 (phi1d[index_later] < 0)) |
                ((phi1d[index] < 0) &
                 (phi1d[index_later] > 0))
        )

        X_zerocrossings = X.loc[mask].copy()
        return X_zerocrossings

    @staticmethod
    def calculate_amplitudes(X_zerocrossings):

        #X_amplitudes = pd.DataFrame()
        #for i in range(len(X_zerocrossings) - 1):
        #    s1 = X_zerocrossings.iloc[i]
        #    s2 = X_zerocrossings.iloc[i + 1]
        #
        #    amplitude = (s2 - s1).abs()
        #    amplitude.name = s2.name - s1.name
        #    X_amplitudes = X_amplitudes.append(amplitude)

        X_amplitudes = X_zerocrossings.copy()
        X_amplitudes['phi']=2*X_zerocrossings['phi'].abs()  # Double amplitude!

        return X_amplitudes

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
        ax.set_title('Peaks')

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

    def plot_fit(self, ax=None):

        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig,ax = plt.subplots()

        df = self.predict(X=self.X)
        self.X.plot(y=self.phi_key, ax=ax, label='Model test')
        df.plot(y='phi', ax=ax, label='Prediction')
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(self.phi_key)





