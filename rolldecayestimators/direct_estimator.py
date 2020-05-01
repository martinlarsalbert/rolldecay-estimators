"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.utils.validation import check_is_fitted
import inspect
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from rolldecayestimators import measure as measure
from sklearn.metrics import r2_score

from rolldecayestimators.substitute_dynamic_symbols import lambdify
from rolldecayestimators.symbols import *
from rolldecayestimators.estimator import RollDecay


class DirectEstimator(RollDecay):
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

    fit_method : str, default='integration'
        The fitting method could be either 'integration' or 'derivation'
        'integration' means that the diff equation is solved with ODE integration.
        The curve_fit runs an integration for each set of parameters to get the best fit.

        'derivation' means that the diff equation os solved by first calculating the 1st and 2nd derivatives numerically.
        The derivatives are then inserted into the diff equation and the best fit is found.


    """

    # Defining the diff equation for this estimator:
    rhs = -phi_dot_dot / (omega0 ** 2) - 2 * zeta / omega0 * phi_dot - d * sp.Abs(phi_dot) * phi_dot / (omega0 ** 2)
    roll_diff_equation = sp.Eq(lhs=phi, rhs=rhs)
    acceleration = sp.Eq(lhs=phi, rhs=sp.solve(roll_diff_equation, phi.diff().diff())[0])
    functions = {'acceleration':lambdify(acceleration.rhs)}

    def simulate(self, t :np.ndarray, phi0 :float, phi1d0 :float,omega0:float, zeta:float, d:float)->pd.DataFrame:
        """
        Simulate a roll decay test using the quadratic method.
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param zeta:linear roll damping [-]
        :param d:quadratic roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """
        parameters={
            'omega0':omega0,
            'zeta':zeta,
            'd':d,
        }
        return self._simulate(t=t, phi0=phi0, phi1d0=phi1d0, parameters=parameters)


    def calculate_amplitudes_and_damping(self):
        X_interpolated = measure.sample_increase(X=self.X)
        self.X_zerocrossings = measure.get_zerocrossings(X=X_interpolated)
        X_amplitudes = measure.calculate_amplitudes(X_zerocrossings=self.X_zerocrossings)
        self.X_amplitudes = self.calculate_damping(X_amplitudes=X_amplitudes)
        T0 = 2*self.X_amplitudes.index
        self.X_amplitudes['omega0'] = 2 * np.pi/T0


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


    def measure_error(self, X):
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

        error = self.measure_error(X=X)

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

    def plot_fft(self, ax=None):
        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig,ax = plt.subplots()

        frequencies, dft = self.fft(self.X['phi'])
        omega=2*np.pi*frequencies

        omega0 = self.fft_omega0(frequencies=frequencies, dft=dft)
        index = np.argmax(np.abs(dft))
        ax.plot(omega, dft)
        ax.plot(omega0,dft[index],'ro')
