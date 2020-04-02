import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from rolldecayestimators.direct_estimator_improved import DirectEstimatorImproved
import matplotlib.pyplot as plt

@pytest.mark.skip('Fix this later, maybe')
def test_fit_simualtion(df_roll_decay):

    direct_estimator = DirectEstimatorImproved()

    X = df_roll_decay
    X['phi2d'] = np.gradient(X['phi1d'].values, X.index.values)

    direct_estimator.fit(X=X)
    X_pred = direct_estimator.predict(X=X)
    fig, ax = plt.subplots()
    X.plot(y='phi', ax=ax, label='actual')
    X_pred.plot(y='phi', ax=ax, label='prediction')
    plt.show()

    assert_almost_equal(X['phi'].values, X_pred['phi'].values, decimal=3)
    assert direct_estimator.score(X) > 0.999


