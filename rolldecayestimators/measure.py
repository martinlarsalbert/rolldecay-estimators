import numpy as np
import pandas as pd
from rolldecayestimators import lambdas

def sample_increase(X, increase=5):
    N = len(X) * increase
    t_interpolated = np.linspace(X.index[0], X.index[-1], N)
    X_interpolated = pd.DataFrame(index=t_interpolated)

    for key, values in X.items():
        X_interpolated[key] = np.interp(t_interpolated, values.index, values)

    return X_interpolated

def get_peaks(X:pd.DataFrame, key='phi1d')->pd.DataFrame:
    """
    Find the peaks in the signal by finding zero roll angle velocity

    Parameters
    ----------
    X
        DataFrame with roll signal as "phi"
    key = 'phi1d'

    Returns
    -------
    Dataframe with rows from X where phi1d is close to 0.

    """

    phi1d = np.array(X[key])

    index = np.arange(0, len(X.index))
    index_later = np.roll(index, shift=-1)
    index_later[-1] = index[-1]
    mask = (
            ((phi1d[index] > 0) &
             (phi1d[index_later] < 0)) |
            ((phi1d[index] < 0) &
             (phi1d[index_later] > 0))
    )

    index_first = index[mask]
    index_second = index[mask] + 1

    # y = m + k*x
    # k = (y2-y1)/(x2-x1)
    # m = y1 - k*x1
    # y = 0 --> x = -m/k
    X_1 = X.iloc[index_first].copy()
    X_2 = X.iloc[index_second].copy()
    rows, cols = X_1.shape

    x1 = np.array(X_1.index)
    x2 = np.array(X_2.index)
    y1 = np.array(X_1['phi1d'])
    y2 = np.array(X_2['phi1d'])
    k = (y2 - y1) / (x2 - x1)
    m = y1 - k * x1
    x = -m / k

    X_1 = np.array(X_1)
    X_2 = np.array(X_2)

    factor = (x - x1) / (x2 - x1)
    factor = np.tile(factor, [cols, 1]).T
    X_zero = X_1 + (X_2 - X_1) * factor

    X_zerocrossings = pd.DataFrame(data=X_zero, columns=X.columns, index=x)

    return X_zerocrossings

def calculate_amplitudes(X_zerocrossings):

    X_amplitudes = pd.DataFrame()
    for i in range(len(X_zerocrossings) - 1):
        s1 = X_zerocrossings.iloc[i]
        s2 = X_zerocrossings.iloc[i + 1]

        amplitude = (s2 - s1).abs()
        amplitude.name = (s1.name + s2.name)/2  # mean time
        X_amplitudes = X_amplitudes.append(amplitude)

    X_amplitudes['phi']/=2
    X_amplitudes['phi_a'] = X_amplitudes['phi']

    return X_amplitudes

def calculate_amplitudes_and_damping(X:pd.DataFrame):
    X_interpolated = sample_increase(X=X)
    X_zerocrossings = get_peaks(X=X_interpolated)
    X_amplitudes = calculate_amplitudes(X_zerocrossings=X_zerocrossings)
    X_amplitudes = calculate_damping(X_amplitudes=X_amplitudes)
    T0 = 2*X_amplitudes.index
    X_amplitudes['omega0'] = 2 * np.pi/np.gradient(T0)
    #X_amplitudes['time'] = np.cumsum(X_amplitudes.index)
    return X_amplitudes

def calculate_damping(X_amplitudes):

    df_decrements = pd.DataFrame()

    for i in range(len(X_amplitudes) - 2):
        s1 = X_amplitudes.iloc[i]
        s2 = X_amplitudes.iloc[i + 2]

        decrement = s1 / s2
        decrement.name = s1.name
        df_decrements = df_decrements.append(decrement)

    df_decrements['zeta_n'] = 1 / (2 * np.pi) * np.log(df_decrements['phi'])

    X_amplitudes_new = X_amplitudes.copy()
    X_amplitudes_new = X_amplitudes_new.iloc[0:-1].copy()
    X_amplitudes_new['zeta_n'] = df_decrements['zeta_n'].copy()
    X_amplitudes_new['B_n'] = 2*X_amplitudes_new['zeta_n']  # [Nm*s]

    return X_amplitudes_new


def fft_omega0(frequencies, dft):

    index = np.argmax(dft)
    natural_frequency = frequencies[index]
    omega0 = 2 * np.pi * natural_frequency
    return omega0

def fft(series):
    """
    FFT of a series
    Parameters
    ----------
    series

    Returns
    -------

    """

    signal = series.values
    time = series.index

    dt = np.mean(np.diff(time))
    #n = 11*len(time)
    n = 50000
    frequencies = np.fft.rfftfreq(n=n, d=dt) # [Hz]

    dft = np.abs(np.fft.rfft(signal, n=n))

    return frequencies, dft


def linearized_matrix(df_rolldecay, df_ikeda, phi_as = np.deg2rad(np.linspace(1,10,10)), g=9.81, rho=1000,
                      do_hatify=True, suffixes=('','_ikeda')):
    """
    Calculate B_e equivalent linearized damping for a range of roll amplitudes for both model tests and simplified ikeda.

    Parameters
    ----------
    df_rolldecay
    df_ikeda
    phi_as

    Returns
    -------

    """


    df = pd.DataFrame()

    for phi_a in phi_as:
        df_ = linearize(phi_a=phi_a, df_rolldecay=df_rolldecay, df_ikeda=df_ikeda, g=g, rho=rho, do_hatify=do_hatify,
                        suffixes=suffixes)
        df_['phi_a']=phi_a
        df =df.append(df_, ignore_index=True)

    return df


def linearize_si(phi_a, df_ikeda, components = ['B_44', 'B_F', 'B_W', 'B_E', 'B_BK', 'B_L'], do_hatify=True):
    """
    Calculate the equivalent linearized damping B_e

    Parameters
    ----------
    phi_a
    df_ikeda
    g
    rho

    Returns
    -------

    """

    df_ikeda = df_ikeda.copy()

    for component in components:
        new_key = '%s_e' % component
        B1_key = '%s_1' % component
        B2_key = '%s_2' % component

        df_ikeda[new_key] = lambdas.B_e_lambda(B_1=df_ikeda[B1_key],
                                               B_2=df_ikeda[B2_key],
                                               omega0=df_ikeda['omega0'],
                                               phi_a=phi_a)

    if do_hatify:
        df_ikeda['B_e'] = df_ikeda['B_44_e']
    else:
        df_ikeda['B_e_hat'] = df_ikeda['B_44_hat_e']

    return df_ikeda

def hatify(df_ikeda, g=9.81, rho=1000, components = ['B','B_44', 'B_F', 'B_W', 'B_E', 'B_BK', 'B_L']):

    df_ikeda=df_ikeda.copy()
    new_keys = ['%s_e' % key for key in components]
    new_hat_keys = ['%s_e_hat' % key for key in components]

    Disp = np.tile(df_ikeda['Disp'],[len(components),1]).T
    beam = np.tile(df_ikeda['beam'],[len(components),1]).T

    df_ikeda[new_hat_keys] = lambdas.B_e_hat_lambda(B_e=df_ikeda[new_keys],
                                                   Disp=Disp,
                                                   beam=beam,
                                                   g=g, rho=rho)

    df_ikeda['B_e_hat'] = df_ikeda['B_44_e_hat']

    return df_ikeda


def linearize_model_test(phi_a, df_rolldecay, g=9.81, rho=1000):
    """
    Calculate the equivalent linearized damping B_e

    Parameters
    ----------
    phi_a
    df_rolldecay
    g
    rho

    Returns
    -------

    """

    df_rolldecay = df_rolldecay.copy()

    df_rolldecay['B_e'] = lambdas.B_e_lambda(B_1=df_rolldecay['B_1'],
                                             B_2=df_rolldecay['B_2'],
                                             omega0=df_rolldecay['omega0'],
                                             phi_a=phi_a)

    df_rolldecay['B_e_hat'] = lambdas.B_e_hat_lambda(B_e=df_rolldecay['B_e'],
                                                     Disp=df_rolldecay['Disp'],
                                                     beam=df_rolldecay['beam'],
                                                     g=g, rho=rho)

    return df_rolldecay

def linearize(phi_a:float, df_rolldecay:pd.DataFrame, df_ikeda:pd.DataFrame, g=9.81, rho=1000,
              components = ['B_44', 'B_F', 'B_W', 'B_E', 'B_BK', 'B_L'], do_hatify=True, suffixes=('','_ikeda')):

    if not do_hatify:
        components = ['%s_hat' % key for key in components]

    df_rolldecay = linearize_model_test(phi_a=phi_a, df_rolldecay=df_rolldecay, g=g, rho=rho)
    df_ikeda = linearize_si(phi_a=phi_a, df_ikeda=df_ikeda, components=components, do_hatify=do_hatify)

    if do_hatify:
        df_ikeda = hatify(df_ikeda=df_ikeda,g=g, rho=rho, components=components)


    df_compare = pd.merge(left=df_rolldecay, right=df_ikeda, how='inner', left_index=True, right_index=True,
                          suffixes=suffixes)
    return df_compare
