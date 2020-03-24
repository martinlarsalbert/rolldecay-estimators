"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import inspect
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#from rolldecayestimators.simulation import simulate
from rolldecayestimators import measure as measure
from sklearn.metrics import r2_score

from rolldecayestimators.substitute_dynamic_symbols import lambdify
from rolldecayestimators.symbols import *

# Defining the diff equation for this estimator:
rhs = -phi_dot_dot/(omega0**2) - 2*zeta/omega0*phi_dot - d*sp.Abs(phi_dot)*phi_dot/(omega0**2)
roll_diff_equation = sp.Eq(lhs=phi,rhs=rhs)
acceleration = sp.Eq(lhs=phi, rhs=sp.solve(roll_diff_equation, phi.diff().diff())[0])
calculate_acceleration = lambdify(acceleration.rhs)


class DirectEstimator(BaseEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    bounds : dict, default=None
        Boundaries for the parameters expressed as dict:
        Ex: {
            'zeta':(-np.inf, np.inf),
            'd':(0,42),
            }

    """

    def __init__(self, maxfev = 4000, bounds={}, ftol=10**-10, p0={}):
        self.is_fitted_ = False

        self.phi_key = 'phi'  # Roll angle [rad]
        self.phi1d_key = 'phi1d'  # Roll velocity [rad/s]
        self.phi2d_key = 'phi2d'  # Roll acceleration [rad/s2]
        self.maxfev = maxfev
        self.ftol = ftol
        self.boundaries = bounds
        self.p0 = p0

        signature = inspect.signature(self.equation)
        self.parameter_names = list(signature.parameters.keys())[1:]

    def get_inital_guess(self):

        p0 = []
        for key in self.parameter_names:
            p0.append(self.p0.get(key,0.5))

        return p0

    def get_bounds(self):

        minimums = []
        maximums = []

        for key in self.parameter_names:

            boundaries = self.boundaries.get(key,(-np.inf, np.inf))
            assert len(boundaries) == 2
            minimums.append(boundaries[0])
            maximums.append(boundaries[1])

        return [tuple(minimums), tuple(maximums)]

    def __repr__(self):
        if self.is_fitted_:
            parameters = ''.join('%s:%0.3f, '%(key,value) for key,value in self.parameters.items())[0:-1]
            return '%s(%s)' % (self.__class__.__name__,parameters)
        else:
            return '%s' % (self.__class__.__name__)

    @staticmethod
    def equation(df, d, zeta):
        phi_old = df['phi']
        p_old = df['phi1d']
        omega0 = df['omega0']

        phi2d = calculate_acceleration(d=d, omega0=omega0, phi1d=p_old, phi=phi_old, zeta=zeta)
        return phi2d

    def roll_decay_time_step(self, states, t, d, omega0, zeta):
        # states:
        # [phi,phi1d]

        phi_old = states[0]
        p_old = states[1]

        phi1d = p_old
        phi2d = calculate_acceleration(d=d, omega0=omega0, phi1d=p_old, phi=phi_old, zeta=zeta)

        d_states_dt = np.array([phi1d, phi2d])

        return d_states_dt

    def simulate(self, t: np.ndarray, phi0: float, phi1d0: float, omega0: float, d: float, zeta: float) -> pd.DataFrame:
        """
        Simulate a roll decay test using the quadratic method.
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param d: quadratic roll damping [-]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """

        states0 = [phi0, phi1d0]
        args = (
            d,
            omega0,
            zeta,
        )
        states = odeint(func=self.roll_decay_time_step, y0=states0, t=t, args=args)

        df = pd.DataFrame(index=t)

        df['phi'] = states[:, 0]
        df['phi1d'] = states[:, 1]

        return df

    def do_simulation(self, t, phi0, phi1d0):
        return self.simulate(t=t, **self.parameters, phi0=phi0, phi1d0=phi1d0)

    def fit(self, X, y=None, calculate_amplitudes_and_damping=True):
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

        self.X = X.copy()
        if calculate_amplitudes_and_damping:
            self.calculate_amplitudes_and_damping()

            self.X['omega0'] = self.omega0


        popt, pcov = curve_fit(f=self.equation, xdata=self.X, ydata=self.X[self.phi2d_key],  maxfev=self.maxfev, ftol=self.ftol,
                               bounds=self.get_bounds(),
                               p0=self.get_inital_guess())

        parameter_values = list(popt)
        parameters = dict(zip(self.parameter_names, parameter_values))

        self.parameters=parameters

        if not 'omega0' in self.parameters:
            self.parameters['omega0'] = self.omega0

        self.pcov = pcov

        return self

    def calculate_amplitudes_and_damping(self):
        X_interpolated = measure.sample_increase(X=self.X)
        self.X_zerocrossings = measure.get_zerocrossings(X=X_interpolated)
        X_amplitudes = measure.calculate_amplitudes(X_zerocrossings=self.X_zerocrossings)
        self.X_amplitudes = self.calculate_damping(X_amplitudes=X_amplitudes)
        T0 = 2*self.X_amplitudes.index
        self.X_amplitudes['omega0'] = 2 * np.pi/T0

    @property
    def omega0(self):
        """
        Mean natural frequency
        Returns
        -------

        """
        return self.X_amplitudes['omega0'].mean()



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

        df_decrements['zeta_n'] *= 2  # !!! # Todo: Where did this one come from?

        X_amplitudes_new = X_amplitudes.copy()
        X_amplitudes_new = X_amplitudes_new.iloc[0:-1].copy()
        X_amplitudes_new['zeta_n'] = df_decrements['zeta_n'].copy()

        return X_amplitudes_new

    def true_and_prediction(self, X=None):

        if X is None:
            X=self.X

        y_true = X[self.phi_key]
        df_sim = self.predict(X)
        y_pred = df_sim[self.phi_key]
        return y_true, y_pred

    def error(self, X):
        y_true, y_pred = self.true_and_prediction(X=X)
        return y_pred - y_true

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

        #sample_weight = np.abs(y_true)
        #return r2_score(y_true=y_true, y_pred=y_pred,sample_weight=sample_weight)
        return r2_score(y_true=y_true, y_pred=y_pred)

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

        signature = inspect.signature(self.simulate)
        parameters = list(signature.parameters.keys())[1:]

        df_sim = self.do_simulation(t=X.index, phi0=phi0, phi1d0=phi1d0)

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
        d = self.parameters.get('d',0)

        if phi_a is None:
            phi_a = self.X[self.phi_key].abs().mean()

        return zeta + 4/(3*np.pi)*d*phi_a

    def plot_fit(self, ax=None, model_test=True,**kwargs):

        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig,ax = plt.subplots()

        df = self.predict(X=self.X)
        df['phi_deg'] = np.rad2deg(df['phi'])

        if model_test:
            X = self.X.copy()
            X['phi_deg'] = np.rad2deg(X['phi'])
            X.plot(y='phi_deg', ax=ax, label='Model test')

        df.plot(y='phi_deg', ax=ax, label=self.__repr__(),**kwargs)

        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$\Phi$ [deg]')

    def plot_error(self,X=None, ax=None, **kwargs):
        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig, ax = plt.subplots()

        error = self.error(X=X)

        ax.plot(self.X.index, error, label=self.__repr__(), **kwargs)
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('error: phi_pred - phi_true [rad]')

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




