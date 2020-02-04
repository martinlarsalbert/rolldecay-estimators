import numpy as np
import pandas as pd

def sample_increase(X):
    N = len(X) * 10
    t_interpolated = np.linspace(X.index[0], X.index[-1], N)
    X_interpolated = pd.DataFrame(index=t_interpolated)

    for key, values in X.items():
        X_interpolated[key] = np.interp(t_interpolated, values.index, values)

    return X_interpolated

def get_zerocrossings(X,key='phi1d'):

    phi1d = np.array(X[key])
    #ts = np.mean(np.diff(X.index))
    #fs = 1/ts
    #cutoff = fs/1000  # ToDo: Verify this assumption
    #phi1d = lowpass_filter(data = phi1d, fs=fs, cutoff=cutoff, order=1)  # Run lowpass filter to remove noice

    index = np.arange(0, len(X.index))
    index_later = np.roll(index, shift=-1)
    index_later[-1] = index[-1]
    mask = (
            ((phi1d[index] > 0) &
             (phi1d[index_later] < 0)) |
            ((phi1d[index] < 0) &
             (phi1d[index_later] > 0))
    )

    X_zerocrossings = X.loc[mask].copy()
    return X_zerocrossings

def calculate_amplitudes(X_zerocrossings):

    X_amplitudes = pd.DataFrame()
    for i in range(len(X_zerocrossings) - 1):
        s1 = X_zerocrossings.iloc[i]
        s2 = X_zerocrossings.iloc[i + 1]

        amplitude = (s2 - s1).abs()
        amplitude.name = s2.name - s1.name
        X_amplitudes = X_amplitudes.append(amplitude)

    #X_amplitudes = X_zerocrossings.copy()
    #X_amplitudes['phi']=2*X_zerocrossings['phi'].abs()  # Double amplitude!

    return X_amplitudes