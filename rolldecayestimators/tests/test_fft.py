import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rolldecayestimators.measure import fft,fft_omega0
from numpy.testing import assert_almost_equal

omega0=2.0
T0 = 2*np.pi/omega0
A = 1.0

@pytest.fixture
def signal():
    t = np.arange(0,10.5*T0,0.01)
    y = A*np.sin(omega0*t)
    s = pd.Series(data=y, index=t)
    yield s

@pytest.fixture
def signal2():
    t = np.arange(0,5.0*T0,0.01)
    y = A*np.sin(omega0*t)
    s = pd.Series(data=y, index=t)
    yield s

def test_fft(signal):

    fig, ax = plt.subplots()
    signal.plot(ax=ax)
    plt.show()

    frequencies, dft = fft(signal)

    index = np.argmax(dft)
    natural_frequency = frequencies[index]
    dft_predict = dft[index]
    omega0_predict = 2 * np.pi * natural_frequency


    fig,ax = plt.subplots()
    ax.plot(2*np.pi*frequencies,dft, '.-', label='dft')
    ax.plot(omega0_predict, dft_predict, 'ro', label='omega_0=%f'%omega0_predict)
    ax.set_xlim(0,omega0_predict+5)
    ax.legend()

    plt.show()

#@pytest.mark.skip('Fix this one...')
def test_fft_omega0(signal):

    frequencies, dft = fft(signal)
    omega0_predict = fft_omega0(frequencies=frequencies, dft=dft)
    assert_almost_equal(omega0_predict, omega0, decimal=2)

def test_fft_omega0_even(signal2):

    frequencies, dft = fft(signal2)
    omega0_predict = fft_omega0(frequencies=frequencies, dft=dft)
    assert_almost_equal(omega0_predict, omega0, decimal=2)

def signal_generator(t,omega0):
    y = A*np.sin(omega0*t)
    s = pd.Series(data=y, index=t)
    return s

def test_fft_omega0_even_many():

    omega0s = np.linspace(0.1,6,10)

    for omega0_ in omega0s:
        T0_ = 2 * np.pi / omega0_
        t = np.arange(0, 10.5*T0_, 0.01)

        s = signal_generator(t=t, omega0=omega0_)

        frequencies, dft = fft(s)
        omega0_predict = fft_omega0(frequencies=frequencies, dft=dft)
        assert_almost_equal(omega0_predict, omega0_, decimal=2)

#@pytest.mark.skip('Fix this one...')
def test_fft_omega0_many():

    omega0s = np.linspace(0.1,6,10)

    for omega0_ in omega0s:
        T0_ = 2 * np.pi / omega0_
        t = np.arange(0, 10.5*T0_, 0.01)

        s = signal_generator(t=t, omega0=omega0_)

        frequencies, dft = fft(s)
        omega0_predict = fft_omega0(frequencies=frequencies, dft=dft)
        assert_almost_equal(omega0_predict, omega0_, decimal=2)


