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

def calculate_variation(df, catch_error=False, limit_inputs=False, verify_input=True):
    result = df.apply(func=calculate, catch_error=catch_error, limit_inputs=limit_inputs, verify_input=verify_input, axis=1)
    return result

def plot_variation(ship, key='lpp', changes=None, ax=None, catch_error=False, plot_change_factor=True):

    if changes is None:
        N = 30
        changes = np.linspace(0.5, 1.5, N)

    df = variate_ship(ship=ship, key=key, changes=changes)
    result = calculate_variation(df=df, catch_error=catch_error)
    result[key] = df[key].copy()

    ax = _plot_result(ship=ship, result=result, key=key, changes=changes, ax=ax, plot_change_factor=plot_change_factor)
    return ax

def _plot_result(ship, result, key, changes, plot_change_factor=True, ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    if plot_change_factor:
        result['change factor'] = changes
        result.plot(x='change factor', ax=ax)
    else:
        result.plot(x=key, ax=ax)

    ax.set_title('Variation of %s: %0.3f' % (key, ship[key]))
    return ax

def calculate(row, catch_error=False, limit_inputs=False, verify_input=True, **kwargs):
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
    V = row.V

    s = pd.Series()
    try:
        B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT, BLHAT = calculate_roll_damping(LPP, Beam, CB, CMID, OG, PHI, lBK, bBK,
                                                                    OMEGA, DRAFT, V=V ,limit_inputs=limit_inputs,
                                                                            verify_input=verify_input, **kwargs)
    except SimplifiedIkedaInputError:
        if catch_error:
            return s
        else:
            raise

    s['B_44_hat'] = B44HAT
    s['B_F_hat'] = BFHAT
    s['B_W_hat'] = BWHAT
    s['B_E_hat'] = BEHAT
    s['B_BK_hat'] = BBKHAT
    s['B_L_hat'] = BLHAT
    return s