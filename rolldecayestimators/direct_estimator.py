"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import inspect
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from rolldecayestimators.equations_lambdify import calculate_acceleration
from rolldecayestimators.simulation import simulate

class DirectEstimator(BaseEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(self):
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

        popt, pcov = curve_fit(f=self.equation, xdata=X, ydata=X[self.phi2d_key],  maxfev=self.maxfev)
        self.X = X.copy()

        signature = inspect.signature(self.equation)
        parameter_names = list(signature.parameters.keys())[1:]

        parameter_values = list(popt)
        parameters = dict(zip(parameter_names, parameter_values))

        self.parameters =parameters
        self.pcov = pcov

        return self

    @staticmethod
    def equation(df, d, omega0, zeta):
        phi_old = df['phi']
        p_old = df['phi1d']

        phi2d = calculate_acceleration(d=d, omega0=omega0, p_old=p_old, phi_old=phi_old, zeta=zeta)
        return phi2d

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

        X.columns

        signature = inspect.signature(simulate)
        parameters = list(signature.parameters.keys())[1:]

        df_sim = self.do_simulation(t=X.index, phi0=phi0, phi1d0=phi1d0)

        #return np.ones(X.shape[0], dtype=np.int64)
        return df_sim

    def do_simulation(self, t, phi0, phi1d0):
        return simulate(t=t, **self.parameters, phi0=phi0, phi1d0=phi1d0)

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
        d = self.parameters.get('d',0)

        if phi_a is None:
            phi_a = self.X[self.phi_key].abs().mean()

        return zeta + 4/(3*np.pi)*d*phi_a

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


class DirectEstimatorZeta(DirectEstimator):

    def __init__(self, d:float):
        super().__init__()
        self.d = d

    @staticmethod
    def equation(df, omega0, zeta):
        phi_old = df['phi']
        p_old = df['phi1d']
        d = df['d']

        phi2d = calculate_acceleration(d=d, omega0=omega0, p_old=p_old, phi_old=phi_old, zeta=zeta)
        return phi2d

    def fit(self, X, y=None):
        X=X.copy()
        X['d'] = self.d
        super().fit(X=X, y=y)
        self.parameters['d'] = self.d

class DirectEstimatorD(DirectEstimator):

    def __init__(self, zeta:float):
        super().__init__()
        self.zeta = zeta

    @staticmethod
    def equation(df, omega0, d):
        phi_old = df['phi']
        p_old = df['phi1d']
        zeta = df['zeta']

        phi2d = calculate_acceleration(d=d, omega0=omega0, p_old=p_old, phi_old=phi_old, zeta=zeta)
        return phi2d

    def fit(self, X, y=None):
        X=X.copy()
        X['zeta'] = self.zeta
        super().fit(X=X, y=y)
        self.parameters['zeta'] = self.zeta






