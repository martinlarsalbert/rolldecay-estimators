import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rolldecayestimators.simplified_ikeda import calculate_roll_damping, SimplifiedIkedaInputError

def variate_ship(ship, key, changes):
    N = len(changes)
    data = np.tile(ship.values, (N, 1))

    df = pd.DataFrame(data=data, columns=ship.index)
    variations = changes * ship[key]
    df[key] = variations
    df.index = df[key].copy()

    return df

def calculate_variation(df):
    result = df.apply(func=calculate, axis=1)
    return result

def plot_variation(ship, key='lpp', changes=None, ax=None):

    if changes is None:
        N = 30
        changes = np.linspace(0.5, 1.5, N)

    df = variate_ship(ship=ship, key=key, changes=changes)
    result = calculate_variation(df=df)

    ax = _plot_result(ship=ship, result=result, key=key, changes=changes, ax=ax)
    return ax

def _plot_result(ship, result, key, changes, ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    result['change factor'] = changes
    result.plot(x='change factor', ax=ax)
    ax.set_title('Variation of %s: %0.2f' % (key, ship[key]))
    return ax

def calculate(row, catch_error=False):
    LPP = row.lpp
    Beam = row.beam
    DRAFT = row.DRAFT

    PHI = row.phi_max
    lBK = row.BKL
    bBK = row.BKB
    OMEGA = row.omega0
    OG = (-row.kg + DRAFT)
    CB = row.CB
    CMID = row.A0

    s = pd.Series()
    try:
        B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT = calculate_roll_damping(LPP, Beam, CB, CMID, OG, PHI, lBK, bBK,
                                                                     OMEGA, DRAFT)
    except SimplifiedIkedaInputError:
        if catch_error:
            return s
        else:
            raise

    s['B44HAT'] = B44HAT
    s['BFHAT'] = BFHAT
    s['BWHAT'] = BWHAT
    s['BEHAT'] = BEHAT
    s['BBKHAT'] = BBKHAT
    return s