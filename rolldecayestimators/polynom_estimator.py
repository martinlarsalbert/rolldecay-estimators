from sklearn.metrics import r2_score
import pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.set_option("display.max_columns", None)
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import os.path

from rolldecayestimators.substitute_dynamic_symbols import lambdify,run


from sklearn.linear_model import LinearRegression
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
from rolldecayestimators import symbols
import dill
dill.settings['recurse'] = True

class Polynom(BaseEstimator):
    """ Estimator that wrappes a model pipline and convert it to a SymPy polynomial expression

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    model  : sklearn.pipeline.Pipeline,
        The model should contain:
        model['polynomial_feature'] : sklearn.feature_selection.SelectKBest
        model['variance_treshold'] : sklearn.feature_selection.VarianceThreshold
        model['select_k_best'] : sklearn.feature_selection.SelectKBest

    columns : list
        a list with the column names from the original dataframe.

    y_symbol : sympy.Symbol
        y_symbol = poly(....)

    """


    def __init__(self, model:Pipeline, columns:list, y_symbol:sp.Symbol):

        self.polynomial_features = model['polynomial_feature']
        self.variance_treshold = model['variance_treshold']
        self.select_k_best = model['select_k_best']

        self.feature_names = np.array(self.polynomial_features.get_feature_names())
        self.polynomial_regression = LinearRegression()
        self.columns = columns
        self.feature_eqs = self.define_feature_equations()
        self.y_symbol = y_symbol

    def fit(self, X, y):
        self.X = X
        result = self.polynomial_regression.fit(X=self.good_X(X), y=y)
        self.equation = self.get_equation()
        self.lamda = lambdify(self.equation.rhs)
        return result

    def score(self, X, y):
        y_pred = self.predict(X=X)
        return r2_score(y_true=y, y_pred=y_pred)

    def predict(self, X):

        if isinstance(X,dict):
            return run(self.lamda, X)

        if isinstance(X,pd.Series):
            return run(self.lamda, X)

        if not isinstance(X, pd.DataFrame):
            assert X.shape[1] == len(self.columns)
            X = pd.DataFrame(data=X, columns=self.columns)

        return run(self.lamda, X)

    @property
    def good_index(self):
        feature_names_index = pd.Series(self.feature_names)
        mask_treshold = self.variance_treshold.get_support()
        mask_select_k_best = self.select_k_best.get_support()
        return feature_names_index[mask_treshold][mask_select_k_best]

    def good_X(self, X):
        X2 = self.polynomial_features.transform(X)
        return X2[:, self.good_index.index]  # Only the good stuff

    def define_sympy_symbols(self):
        self.sympy_symbols = {key: getattr(symbols, key) for key in self.columns}

    def define_parameters(self):
        self.parameters = [self.sympy_symbols[key] for key in self.columns]

    def define_feature_equations(self):

        self.define_sympy_symbols()
        self.define_parameters()

        xs = {'x%i' % i: name for i, name in enumerate(self.columns)}
        xs_sympy = {sp.Symbol(key): self.sympy_symbols[value] for key, value in xs.items()}

        feature_eqs = [1.0, ]

        for feature in self.feature_names[1:]:
            s_eq = feature

            s_eq = s_eq.replace(' ', '*')
            s_eq = s_eq.replace('^', '**')

            sympy_eq = parse_expr(s_eq)
            sympy_eq = sympy_eq.subs(xs_sympy)
            feature_eqs.append(sympy_eq)

        return np.array(feature_eqs)

    @property
    def good_feature_equations(self):
        return self.feature_eqs[self.good_index.index]

    def get_equation(self):
        rhs = 0
        for good_feature_equation, coeff in zip(self.good_feature_equations,
                                                self.polynomial_regression.coef_):
            rhs += coeff * good_feature_equation
        rhs += self.polynomial_regression.intercept_

        return sp.Eq(self.y_symbol, rhs)

    def save(self, file_path:str):

        if not os.path.splitext(file_path)[-1]:
            file_path+='.sym'

        dill.dump(self, open(file_path, 'wb'))

    @classmethod
    def load(cls, file_path:str):
        with open(file_path, 'rb') as file:
            polynom = dill.load(file)

        polynom.equation = polynom.get_equation()
        polynom.lamda = lambdify(polynom.equation.rhs)

        return polynom