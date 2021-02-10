import inspect
from scipy.optimize import least_squares
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd

from rolldecayestimators.substitute_dynamic_symbols import lambdify
from rolldecayestimators.symbols import *
from rolldecayestimators.measure import fft, fft_omega0

class FitError(Exception):pass

class RollDecay(BaseEstimator):

    # Defining the diff equation for this estimator:
    rhs = -phi_dot_dot/(omega0**2) - 2*zeta/omega0*phi_dot
    roll_diff_equation = sp.Eq(lhs=phi, rhs=rhs)
    acceleration = sp.Eq(lhs=phi, rhs=sp.solve(roll_diff_equation, phi.diff().diff())[0])
    functions = {
        'acceleration':lambdify(acceleration.rhs)
    }

    def __init__(self, ftol=1e-09, maxfev=100000, bounds={}, p0={}, fit_method='derivation', omega_regression=True, omega0=None):
        self.is_fitted_ = False

        self.phi_key = 'phi'  # Roll angle [rad]
        self.phi1d_key = 'phi1d'  # Roll velocity [rad/s]
        self.phi2d_key = 'phi2d'  # Roll acceleration [rad/s2]
        self.y_key = self.phi2d_key
        self.boundaries = bounds
        self.p0 = p0
        self.maxfev=maxfev
        self.ftol=ftol
        self.set_fit_method(fit_method=fit_method)
        self.omega_regression = omega_regression
        self.assert_success = True
        self._omega0 = omega0

    @classmethod
    def load(cls,data:{}, X=None):
        """
        Load data and parameters from an existing fitted estimator

        Parameters
        ----------
        data : dict
            Dict containing data for this estimator such as parameters
        X : pd.DataFrame
            DataFrame containing the measurement that this estimator fits (optional).
        Returns
        -------
        estimator
            Loaded with parameters from data and maybe also a loaded measurement X
        """
        return cls._load(data=data, X=X)

    @classmethod
    def _load(cls,data:{}, X=None):
        """
        Load data and parameters from an existing fitted estimator

        Parameters
        ----------
        data : dict
            Dict containing data for this estimator such as parameters
        X : pd.DataFrame
            DataFrame containing the measurement that this estimator fits (optional).
        Returns
        -------
        estimator
            Loaded with parameters from data and maybe also a loaded measurement X
        """
        estimator = cls()
        estimator.load_data(data=data)
        estimator.load_X(X=X)
        return estimator

    def load_data(self,data:{}):
        parameter_names = self.parameter_names
        missing = list(set(parameter_names) - set(data.keys()))
        if len(missing) > 0:
            raise ValueError('The following parameters are missing in data:%s' % missing)

        parameters = {key: value for key, value in data.items() if key in parameter_names}
        self.parameters = parameters
        self.is_fitted_ = True

    def load_X(self, X=None):
        if isinstance(X, pd.DataFrame):
            self.X=X

    def set_fit_method(self,fit_method):
        self.fit_method = fit_method

        if self.fit_method == 'derivation':
            self.y_key=self.phi2d_key
        elif self.fit_method == 'integration':
            self.y_key=self.phi_key
        else:
            raise ValueError('Unknown fit_mehod:%s' % self.fit_method)

    def __repr__(self):
        if self.is_fitted_:
            parameters = ''.join('%s:%0.3f, '%(key,value) for key,value in sorted(self.parameters.items()))[0:-1]
            return '%s(%s)' % (self.__class__.__name__,parameters)
        else:
            return '%s' % (self.__class__.__name__)

    @property
    def calculate_acceleration(self):
        return self.functions['acceleration']

    @property
    def parameter_names(self):
        signature = inspect.signature(self.calculate_acceleration)

        remove = [self.phi_key, self.phi1d_key]
        if not self.omega_regression:
            remove.append('omega0')

        return list(set(signature.parameters.keys()) - set(remove))

    @staticmethod
    def error(x, self, xs, ys):
        #return np.sum((ys - self.estimator(x, xs))**2)
        return ys-self.estimator(x, xs)


    def estimator(self, x, xs):
        self.parameters = {key: x for key, x in zip(self.parameter_names, x)}

        if not self.omega_regression:
            self.parameters['omega0'] = self.omega0

        if self.fit_method=='derivation':
            self.parameters['phi'] = xs[self.phi_key]
            self.parameters['phi1d'] = xs[self.phi1d_key]
            return self.estimator_acceleration(parameters=self.parameters)
        elif self.fit_method=='integration':
            t = xs.index
            phi0=xs.iloc[0][self.phi_key]
            phi1d0 = xs.iloc[0][self.phi1d_key]

            return self.estimator_integration(t=t, phi0=phi0, phi1d0=phi1d0)
        else:
            raise ValueError('Unknown fit_mehod:%s' % self.fit_method)

    def estimator_acceleration(self,parameters):
        acceleration = self.calculate_acceleration(**parameters)
        return acceleration

    def estimator_integration(self, t, phi0, phi1d0):

        try:
            df = self.simulate(t=t, phi0=phi0, phi1d0=phi1d0)
        except:
            df = pd.DataFrame(index=t)
            df['phi']=np.inf
            df['phi1d']=np.inf

        return df[self.y_key]

    def fit(self, X, y=None, **kwargs):
        self.X = X.copy()

        kwargs = {'self': self,
                  'xs': X,
                  'ys': X[self.y_key]}


        if self.fit_method=='integration':
            self.result = least_squares(fun=self.error, x0=self.initial_guess, kwargs=kwargs, bounds=self.bounds,
                                    ftol=self.ftol, max_nfev=self.maxfev, loss='soft_l1', f_scale=0.1)
        else:
            self.result = least_squares(fun=self.error, x0=self.initial_guess, kwargs=kwargs,
                                    ftol=self.ftol, max_nfev=self.maxfev, method='lm')

        if self.assert_success:
            if not self.result['success']:
                raise FitError(self.result['message'])

        self.parameters = {key: x for key, x in zip(self.parameter_names, self.result.x)}

        if not self.omega_regression:
            self.parameters['omega0'] = self.omega0

        self.is_fitted_ = True

    def simulate(self, t, phi0, phi1d0)->pd.DataFrame:

        states0 = [phi0, phi1d0]

        #states = odeint(self.roll_decay_time_step, y0=states0, t=t, args=(self,parameters))
        #df[self.phi_key] = states[:, 0]
        #df[self.phi1d_key] = states[:, 1]

        t_ = t-t[0]
        t_span = [t_[0], t_[-1]]
        self.simulation_result = solve_ivp(fun=self.roll_decay_time_step, t_span=t_span, y0=states0, t_eval=t_,
                                           )
        if not self.simulation_result['success']:
            raise ValueError('Simulation failed')

        df = pd.DataFrame(index=t)
        df[self.phi_key] = self.simulation_result.y[0, :]
        df[self.phi1d_key] = self.simulation_result.y[1, :]
        p_old = df[self.phi1d_key]
        phi_old = df[self.phi_key]
        df[self.phi2d_key] = self.calculate_acceleration(phi1d=p_old, phi=phi_old, **self.parameters)

        return df

    def roll_decay_time_step(self,t,states):
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
        t = np.array(X.index)
        return self.simulate(t=t, phi0=phi0, phi1d0=phi1d0)


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

        return r2_score(y_true=y_true, y_pred=y_pred)

    def true_and_prediction(self, X=None):

        if X is None:
            X=self.X

        y_true = X[self.phi_key]
        df_sim = self.predict(X)
        y_pred = df_sim[self.phi_key]
        return y_true, y_pred

    @property
    def bounds(self):

        minimums = []
        maximums = []

        for key in self.parameter_names:

            boundaries = self.boundaries.get(key,(-np.inf, np.inf))
            assert len(boundaries) == 2
            minimums.append(boundaries[0])
            maximums.append(boundaries[1])

        return [tuple(minimums), tuple(maximums)]

    @property
    def initial_guess(self):
        p0 = []
        for key in self.parameter_names:
            p0.append(self.p0.get(key,0.5))

        return p0

    @property
    def omega0(self):
        """
        Mean natural frequency
        Returns
        -------

        """

        if not self._omega0 is None:
            return self._omega0

        frequencies, dft = fft(self.X['phi'])
        omega0 = fft_omega0(frequencies=frequencies, dft=dft)

        return omega0


    def result_for_database(self, meta_data={}, score=True):
        check_is_fitted(self, 'is_fitted_')

        s = {}
        s.update(self.parameters)
        if score:
            s['score'] = self.score(X=self.X)

        if not self.X is None:
            s['phi_start'] = self.X.iloc[0]['phi']
            s['phi_stop'] = self.X.iloc[-1]['phi']

        if hasattr(self,'omega0'):
            s['omega0_fft'] = self.omega0

        self.results = s  # Store it also

        return s



