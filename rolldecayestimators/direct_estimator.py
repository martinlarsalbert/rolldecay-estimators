"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import inspect
from scipy.optimize import curve_fit


class TemplateEstimator(BaseEstimator):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, calculate_acceleration,p0=None):
        self.calculate_acceleration = calculate_acceleration
        self.p0 = p0

    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        # `fit` should always return `self`

        def f_direct(df, **kwargs):
            phi_old = df['phi']
            p_old = df['phi1d']

            phi2d = self.calculate_acceleration(p_old=p_old, phi_old=phi_old,**kwargs)

            return phi2d


        popt, pcov = curve_fit(f=f_direct, xdata=X, ydata=X['phi2d'], p0=self.p0)

        signature = inspect.signature(f_direct)
        parameter_names = list(signature.parameters.keys())[1:]

        parameter_values = list(popt)
        parameters = dict(zip(parameter_names, parameter_values))

        self.parameters =parameters
        self.pcov = pcov

        return self

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
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.ones(X.shape[0], dtype=np.int64)


class RollDecayCutTransformer(BaseEstimator, TransformerMixin):
    """ Rolldecay transformer that cut time series from roll decay test for estimator.

    Parameters
    ----------
    start : float, default=0
        Start of the cut expressed as a portion of max(abs(phi)) [0..1]

    stop : float, default=1
        Stop of the cut expressed as a portion of max(abs(phi)) [0..1]


    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, start=0, stop=1):
        self.start = start
        self.stop = stop

    def fit(self, X, y):
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
        X = check_array(X, accept_sparse=True)

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
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        if (self.start < 0) or (self.start > 1):
            raise ValueError('"start" should be in the intervall 0..1')

        if (self.stop < 0) or (self.stop > 1):
            raise ValueError('"stop" should be in the intervall 0..1')

        if (self.start >= self.stop):
            raise ValueError('"start" cannot be larger or equal to "stop"')

        return X
