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

    @classmethod
    def load(cls, omega0:float, d:float, zeta:float, X=None):
        """
        Load data and parameters from an existing fitted estimator

        Parameters
        ----------
        omega0 :
            roll frequency [rad/s]
        d : nondimensional linear damping
        zeta :
            nondimensional quadratic damping
        X : pd.DataFrame
            DataFrame containing the measurement that this estimator fits (optional).
        Returns
        -------
        estimator
            Loaded with parameters from data and maybe also a loaded measurement X
        """
        data={
            'd':d,
            'zeta':zeta,
            'omega0':omega0,
        }

        return super(cls, cls)._load(data=data, X=X)

    def calculate_amplitudes_and_damping(self):
        if hasattr(self,'X'):
            if not self.X is None:
                self.X_amplitudes=measure.calculate_amplitudes_and_damping(X=self.X)

                if self.is_fitted_:
                    X_pred = self.predict(X=self.X)
                    self.X_pred_amplitudes = measure.calculate_amplitudes_and_damping(X=X_pred)


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
        X_amplitudes_new['B_n'] = 2*X_amplitudes_new['zeta_n']  # [Nm*s]

        return X_amplitudes_new


    def measure_error(self, X):
        y_true, y_pred = self.true_and_prediction(X=X)
        return y_pred - y_true

    def score(self, X=None, y=None, sample_weight=None):
        """
        Return the coefficient of determination R_b^2 of the prediction.

        The coefficient R_b^2 is defined as (1 - u/v), where u is the residual sum of squares
        ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
        A constant model that always predicts the expected value of y, disregarding the input features,
        would get a R_b^2 score of 0.0.

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
            R_b^2 of self.predict(X) wrt. y.

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

    def plot_fit(self, ax=None, include_model_test=True, label=None, **kwargs):

        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig,ax = plt.subplots()

        df = self.predict(X=self.X)
        df['phi_deg'] = np.rad2deg(df['phi'])

        if include_model_test:
            X = self.X.copy()
            X['phi_deg'] = np.rad2deg(X['phi'])
            X.plot(y=r'phi_deg', ax=ax, label='Model test')

        if not label:
            label = 'fit'
        df.plot(y=r'phi_deg', ax=ax, label=label, style='--',**kwargs)

        ax.legend()
        ax.set_xlabel(r'Time [s]')
        ax.set_ylabel(r'$\Phi$ [deg]')

    def plot_error(self,X=None, ax=None, **kwargs):
        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig, ax = plt.subplots()

        error = self.measure_error(X=X)

        ax.plot(self.X.index, error, label=self.__repr__(), **kwargs)
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('error: phi_pred - phi_true [rad]')

    def plot_peaks(self, ax=None, **kwargs):
        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig,ax = plt.subplots()

        #self.X.plot(y='phi', ax=ax)

        self.X_amplitudes.plot(y='phi', ax=ax, **kwargs)

        #ax.plot([np.min(self.X.index),np.max(self.X.index)],[0,0],'m-')
        ax.set_title('Peaks')

    def plot_velocity(self, ax=None):
        check_is_fitted(self, 'is_fitted_')

        if ax is None:
            fig,ax = plt.subplots()

        self.X.plot(y='phi1d', ax=ax)
        self.X_amplitudes.plot(y='phi1d', ax=ax, style='r.')
        ax.plot([np.min(self.X.index), np.max(self.X.index)], [0, 0], 'm-')
        ax.set_title('Velocities')

    def plot_amplitude(self, ax=None, include_model_test=True):

        if ax is None:
            fig,ax = plt.subplots()

        if not hasattr(self,'X_amplitudes'):
            self.calculate_amplitudes_and_damping()

        if include_model_test:
            X_amplitudes=self.X_amplitudes.copy()
            X_amplitudes['phi_a']=np.rad2deg(X_amplitudes['phi_a'])
            X_amplitudes.plot(y='phi_a', style='o', label='Model test', ax=ax)

        if hasattr(self,'X_pred_amplitudes'):
            label = self.__repr__()
            X_pred_amplitudes = self.X_pred_amplitudes.copy()
            X_pred_amplitudes['phi_a'] = np.rad2deg(X_pred_amplitudes['phi_a'])
            X_pred_amplitudes.plot(y='phi_a', label=label, ax=ax)

    def plot_damping(self, ax=None, include_model_test=True,label=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots()

        plot = None
        if include_model_test:
            X_amplitudes = measure.calculate_amplitudes_and_damping(X=self.X)
            X_amplitudes['phi_a'] = np.rad2deg(X_amplitudes['phi_a'])
            plot = X_amplitudes.plot(x='phi_a', y='B_n', style='o', label='Model test', ax=ax, **kwargs)

        if self.is_fitted_:
            if not label:
                label = self.__repr__()

            X_pred = self.predict(X=self.X)
            X_pred_amplitudes = measure.calculate_amplitudes_and_damping(X=X_pred)
            X_pred_amplitudes['phi_a'] = np.rad2deg(X_pred_amplitudes['phi_a'])

            if plot is None:
                color=None
            else:
                line = plot.axes.get_lines()[-1]
                color = line.get_color()

            x = X_pred_amplitudes['phi_a']
            y = X_pred_amplitudes['B_n']
            ax.plot(x, y, color=color, label=label, **kwargs, lw=2)

        ax.set_xlabel(r'$\phi_a$ [deg]')
        ax.set_ylabel(r'$B$ [Nms]')
        ax.legend()

    def plot_omega0(self,ax=None, include_model_test=True, label=None, **kwargs):

        if ax is None:
            fig,ax = plt.subplots()

        if not hasattr(self, 'X_amplitudes'):
            self.calculate_amplitudes_and_damping()

        plot = None
        if include_model_test:

            X_amplitudes = self.X_amplitudes.copy()
            X_amplitudes['omega0_norm'] = X_amplitudes['omega0']/self.omega0
            X_amplitudes['phi_a_deg'] = np.rad2deg(X_amplitudes['phi_a'])
            plot = X_amplitudes.plot(x='phi_a_deg', y='omega0_norm', style='o', label='Model test', ax=ax, **kwargs)

        if hasattr(self, 'X_pred_amplitudes'):
            if not label:
                label = self.__repr__()

            if plot is None:
                color = None
            else:
                line = plot.axes.get_lines()[-1]
                color = line.get_color()

            x = np.rad2deg(self.X_pred_amplitudes['phi_a'])
            y = self.X_pred_amplitudes['omega0']/self.omega0

            ax.plot(x, y, color=color, label=label, **kwargs, lw=2)

        ax.set_xlabel(r'$\phi_a$ [deg]')
        ax.set_ylabel(r'$\frac{\omega_0^N}{\omega_0}$ [-]')
        ax.legend()

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
