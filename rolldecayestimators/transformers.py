import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression
import pandas as pd

import rolldecayestimators.filters
import rolldecayestimators.measure as measure
from sklearn.metrics import r2_score

class CutTransformer(BaseEstimator, TransformerMixin):
    """ Rolldecay transformer that cut time series from roll decay test for estimator.

    Parameters
    ----------
    phi_max : float, default=np.deg2rad(90)
        Start cutting value is below this value [rad]

    phi_min : float, default=0
        Stop cutting value is when below this value [rad]

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, phi_max=np.deg2rad(90), phi_min=0, phi1d_start_tolerance=0.005):
        self.phi_max = phi_max  # Maximum Roll angle [rad]
        self.phi_min = phi_min  # Minimum Roll angle [rad]
        self.phi_key = 'phi'  # Roll angle [rad]
        self.remove_end_samples = 200  # Remove this many samples from end (funky stuff may happen during end of tests)
        self.phi1d_start_tolerance = phi1d_start_tolerance

    def fit(self, X, y=None):
        """Do the cut

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        #X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        phi = X[self.phi_key]
        if (self.phi_max < phi.abs().min()):
            raise ValueError('"phi_max" is too small')

        if (self.phi_min > phi.abs().max()):
            raise ValueError('"phi_min" is too large')

        if not isinstance(self.remove_end_samples,int):
            raise ValueError('"remove_end_samples" should be integer')

        if self.remove_end_samples<1:
            raise ValueError('"remove_end_samples" > 1')

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        #X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        #if X.shape[1] != self.n_features_:
        #    raise ValueError('Shape of input is different from what was seen'
        #                     'in `fit`')

        #Remove initial part (by removing first to maximums):
        phi = X[self.phi_key]
        index = phi.abs().idxmax()
        X_cut = X.loc[index:].copy()

        if (len(X_cut) > 10*self.remove_end_samples):
            X_cut = X_cut.iloc[0:-self.remove_end_samples]

        phi = X_cut[self.phi_key]
        phi_max_sign = np.sign(phi.loc[index])
        if phi_max_sign == 1:
            index2 = phi.idxmin()
        else:
            index2 = phi.idxmax()

        X_cut = X_cut.loc[index2:].copy()

        X_interpolated = measure.sample_increase(X=X_cut, increase=5)
        X_zerocrossings = measure.get_peaks(X=X_interpolated)
        mask = X_interpolated.index >= X_zerocrossings.index[0]
        X_interpolated = X_interpolated.loc[mask]

        # Remove some large angles at start
        mask = X_zerocrossings['phi'].abs() < self.phi_max
        X_zerocrossings2 = X_zerocrossings.loc[mask].copy()
        if len(X_zerocrossings2) > 0:
            mask2 = X_interpolated.index > X_zerocrossings2.index[0]
            X_interpolated = X_interpolated.loc[mask2]


        # Remove some small angles at end
        mask = X_zerocrossings2['phi'].abs() < self.phi_min
        X_zerocrossings3 = X_zerocrossings2.loc[mask].copy()
        if len(X_zerocrossings3) > 0:
            mask3 = X_interpolated.index < X_zerocrossings3.index[0]
            X_interpolated = X_interpolated.loc[mask3]

        if 'phi1d' in X_cut:
            phi1d_start = np.abs(X_interpolated.iloc[0]['phi1d'])
        
            if phi1d_start > self.phi1d_start_tolerance:
                raise ValueError('Start phi1d exceeds phi1d_start_tolerance (%f > %f)' % (phi1d_start, self.phi1d_start_tolerance) )

        mask = ((X_cut.index >= X_interpolated.index[0]) & (X_cut.index <= X_interpolated.index[-1]))
        X_cut=X_cut.loc[mask].copy()

        return X_cut

class LowpassFilterDerivatorTransformer(BaseEstimator, TransformerMixin):
    """ Rolldecay transformer that lowpass filters the roll signal for estimator.

    Parameters
    ----------
    phi_max : float, default=np.deg2rad(90)
        Start cutting value is below this value [rad]

    phi_min : float, default=0
        Stop cutting value is when below this value [rad]

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, cutoff=0.5, order=5, minimum_score=0.999):
        self.cutoff = cutoff
        self.order = order
        self.phi_key = 'phi'  # Roll angle [rad]
        self.phi_filtered_key = 'phi_filtered'  # Filtered roll angle [rad]
        self.phi1d_key = 'phi1d'  # Roll velocity [rad/s]
        self.phi2d_key = 'phi2d'  # Roll acceleration [rad/s2]
        self.minimum_score = minimum_score

    def fit(self, X, y=None):
        """Do the cut

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        #X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        assert self.score(X=X) > self.minimum_score

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        #X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        #if X.shape[1] != self.n_features_:
        #    raise ValueError('Shape of input is different from what was seen'
        #                     'in `fit`')

        # Lowpass filter the signal:
        self.X = X.copy()
        self.X_filter = X.copy()
        ts = np.mean(np.diff(self.X_filter.index))
        fs = 1 / ts
        self.X_filter[self.phi_filtered_key] = rolldecayestimators.filters.lowpass_filter(data=self.X_filter['phi'],
                                                                                     cutoff=self.cutoff, fs=fs,
                                                                                     order=self.order)

        self.X_filter = self.add_derivatives(X=self.X_filter)

        return self.X_filter

    def plot_filtering(self):

        fig, axes = plt.subplots(nrows=3)

        ax = axes[0]
        self.X.plot(y='phi', ax=ax)
        self.X_filter.plot(y='phi_filtered', ax=ax, style='--')
        ax.legend();

        ax = axes[1]
        self.X_filter.plot(y='phi1d', ax=ax, style='--')
        ax.legend();

        ax = axes[2]
        self.X_filter.plot(y='phi2d', ax=ax, style='--')
        ax.legend();

    def add_derivatives(self, X):
        # Add accelerations:
        assert self.phi_key in X
        X = X.copy()
        X[self.phi1d_key] = np.gradient(X[self.phi_filtered_key].values, X.index.values)
        X[self.phi2d_key] = np.gradient(X[self.phi1d_key].values, X.index.values)
        return X

    def score(self, X, y=None, sample_weight=None):
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
        X_filter = self.transform(X)
        y_true = X[self.phi_key]
        y_pred = X_filter[self.phi_filtered_key]

        return r2_score(y_true=y_true, y_pred=y_pred)

class ScaleFactorTransformer(BaseEstimator, TransformerMixin):
    """ Rolldecay to full scale using scale factor

    Parameters
    ----------
    phi_max : float, default=np.deg2rad(90)
        Start cutting value is below this value [rad]

    phi_min : float, default=0
        Stop cutting value is when below this value [rad]

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.phi1d_key = 'phi1d'  # Roll velocity [rad/s]
        self.phi2d_key = 'phi2d'  # Roll acceleration [rad/s2]

    def fit(self, X, y=None):
        """Do the cut

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        #X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        if pd.isnull(self.scale_factor):
            raise ValueError('Bad scale factor:%s' % self.scale_factor)

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        #X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        #if X.shape[1] != self.n_features_:
        #    raise ValueError('Shape of input is different from what was seen'
        #                     'in `fit`')


        X_scaled = X.copy()
        X_scaled.index*=np.sqrt(self.scale_factor)  # To full scale
        if self.phi1d_key in X:
            X_scaled[self.phi1d_key]/=np.sqrt(self.scale_factor)

        if self.phi2d_key in X:
            X_scaled[self.phi2d_key]/=self.scale_factor


        return X_scaled

class OffsetTransformer(BaseEstimator, TransformerMixin):
    """ Rolldecay remove offset in signal

    Parameters
    ----------
    phi_max : float, default=np.deg2rad(90)
        Start cutting value is below this value [rad]

    phi_min : float, default=0
        Stop cutting value is when below this value [rad]

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self):
        self.phi1d_key = 'phi1d'  # Roll velocity [rad/s]
        self.phi2d_key = 'phi2d'  # Roll acceleration [rad/s2]

    def fit(self, X, y=None):
        """Do the cut

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        #X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        #X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        #if X.shape[1] != self.n_features_:
        #    raise ValueError('Shape of input is different from what was seen'
        #                     'in `fit`')


        X_offset = X.copy()

        #X_interpolated = measure.sample_increase(X=X)
        X_zerocrossings = measure.get_peaks(X=X_offset)

        linear_regression = LinearRegression(fit_intercept=False)
        X_ = np.array([X_zerocrossings.index.values]).transpose()
        linear_regression.fit(X=X_, y=X_zerocrossings['phi'])
        X_zerocrossings['phi0'] = linear_regression.predict(X=X_)
        X_zerocrossings['phi_'] = X_zerocrossings['phi'] - X_zerocrossings['phi0']

        X_2 = np.array([X_offset.index.values]).transpose()
        X_offset['phi0'] = linear_regression.predict(X=X_2)
        X_offset['phi_offset'] = X_offset['phi'].copy()
        X_offset['phi'] = X_offset['phi'] - X_offset['phi0']

        self.X = X_offset

        return X_offset

    def plot_correction_line(self, ax=None):

        check_is_fitted(self, 'n_features_')

        if ax is None:
            fig,ax=plt.subplots()

        self.X_zerocrossings.plot(y='phi', style='ro', ax=ax, label='phi')
        self.X_zerocrossings.plot(y='phi0', style='b-', ax=ax, label='correction line')
        ax.plot([np.min(self.X_zerocrossings.index), np.max(self.X_zerocrossings.index)], [0, 0], 'g-')
        ax.legend()

    def plot(self, ax=None):

        check_is_fitted(self, 'n_features_')

        if ax is None:
            fig,ax=plt.subplots()

        self.X.plot(y='phi_offset', ax=ax, label='old')
        self.X.plot(y='phi', ax=ax, label='corrected')
        ax.plot([np.min(self.X.index), np.max(self.X.index)], [0, 0], 'g-')
        ax.legend()
