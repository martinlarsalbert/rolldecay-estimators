import pytest
import pandas as pd
import os.path
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import VarianceThreshold
import sympy as sp

from rolldecayestimators.polynom_estimator import Polynom
import matplotlib.pyplot as plt

@pytest.fixture
def data():
    # Generate some random regression data:
    X, y = make_regression(n_samples=1000, n_features=1, n_informative=10, noise=10, random_state=1)
    yield X, y


@pytest.fixture
def model(data):

    X=data[0]
    y=data[1]

    polynomial_features = PolynomialFeatures(degree=1)
    variance_treshold = VarianceThreshold()
    linear_regression = LinearRegression()

    select_k_best = SelectKBest(k=1, score_func=f_regression)
    steps = [
        ('polynomial_feature', polynomial_features),
        # ('standard_scaler', standard_scaler),
        ('variance_treshold', variance_treshold),
        ('select_k_best', select_k_best),
        ('linear_regression', linear_regression)
    ]

    model = Pipeline(steps=steps)
    model.fit(X=X, y=y)

    yield model


def test_fit(data, model):

    X=data[0]
    x=X[:,0]
    y=data[1]

    y_symbol = sp.symbols('y')
    polynom = Polynom(model=model, columns=['B_e_hat'], y_symbol=y_symbol)
    polynom.fit(X=X, y=y)

    fig,ax=plt.subplots()
    ax.plot(x,y, '.')
    ax.plot(x, model.predict(X=X), label='model')
    ax.plot(x, polynom.predict(X=X), '--', label='polynom')

    ax.legend()
    plt.show()

@pytest.fixture
def polynom(data, model):

    X = data[0]
    y = data[1]
    y_symbol = sp.symbols('y')
    polynom = Polynom(model=model, columns=['B_e_hat'], y_symbol=y_symbol)
    polynom.fit(X=X, y=y)
    yield polynom

def test_predict_dict(polynom):

    data = {
        'B_e_hat':1,
    }

    polynom.predict(X=data)

def test_predict_series(polynom):

    data = {
        'B_e_hat':1,
    }
    data = pd.Series(data)

    polynom.predict(X=data)

def test_save_load(tmpdir, polynom):

    file_path=os.path.join(str(tmpdir), 'test.sym')
    polynom.save(file_path=file_path)
    polynom2 = Polynom.load(file_path=file_path)
    data = {
        'B_e_hat': 1,
    }

    polynom2.predict(X=data)

