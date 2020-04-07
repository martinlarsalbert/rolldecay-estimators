import pandas as pd
import numpy as np
import sympy as sp

from rolldecayestimators.estimator import RollDecay
from rolldecayestimators.simplified_ikeda import calculate_roll_damping
from rolldecayestimators import equations
from rolldecayestimators import symbols
from rolldecayestimators.substitute_dynamic_symbols import lambdify

class IkedaEstimator(RollDecay):

    eqs = [equations.zeta_equation,
           equations.omega0_equation_linear]
    functions_ikeda = (lambdify(sp.solve(eqs, symbols.A_44, symbols.zeta)[0][1]),
                       lambdify(sp.solve(equations.B44_equation, symbols.B_44)[0]),
                       )

    def __init__(self, lpp:float, TA, TF, beam, BKL, BKB, A0, kg, Volume, gm, rho=1000, g=9.81):
        """
        Estimate a roll decay test using the Simplified Ikeda Method to predict roll damping.
        NOTE! This method is currently only valid for zero speed!

        Parameters
        ----------
        lpp
            Ship perpendicular length [m]
        TA
            Draught aft [m]
        TF
            Draught forward [m]
        beam
            Ship beam [m]
        BKL
            Bilge keel length [m]
        BKB
            Bilge keel height [m]
        A0
            Middship coefficient (A_m/(B*d) [-]
        kg
            Vertical centre of gravity [m]
        Volume
            Displacement of ship [m3]
        gm
            metacentric height [m]
        rho
            Density of water [kg/m3]
        g
            acceleration of gravity [m/s**2]

        For more info see: "rolldecaysestimators/simplified_ikeda.py"
        """
        super().__init__()

        self.lpp=lpp
        self.TA=TA
        self.TF=TF
        self.beam=beam
        self.BKL=BKL
        self.BKB=BKB
        self.A0=A0
        self.kg=kg
        self.Volume=Volume
        self.rho=rho
        self.g=g
        self.gm=gm

    @property
    def zeta_lambda(self):
        return self.functions_ikeda[0]

    @property
    def B44_lambda(self):
        return self.functions_ikeda[1]

    def simulate(self, t :np.ndarray, phi0 :float, phi1d0 :float,omega0:float, zeta:float)->pd.DataFrame:
        """
        Simulate a roll decay test using the quadratic method.
        :param t: time vector to be simulated [s]
        :param phi0: initial roll angle [rad]
        :param phi1d0: initial roll speed [rad/s]
        :param omega0: roll natural frequency[rad/s]
        :param zeta:linear roll damping [-]
        :return: pandas data frame with time series of 'phi' and 'phi1d'
        """
        parameters={
            'omega0':omega0,
            'zeta':zeta,
        }
        return self._simulate(t=t, phi0=phi0, phi1d0=phi1d0, parameters=parameters)

    def fit(self, X, y=None, **kwargs):
        self.X = X

        self.phi_max = np.rad2deg(self.X[self.phi_key].abs().max())  ## Initial roll angle in [deg]

        DRAFT=(self.TA + self.TF) / 2
        omega0=self.omega0
        self.ikeda_parameters = {

            'LPP' : self.lpp,
            'Beam' : self.beam,
            'DRAFT' : DRAFT,

            'PHI' : self.phi_max,
            'lBK' : self.BKL,
            'bBK' : self.BKB,
            'OMEGA' : omega0,
            'OG' : (-self.kg + DRAFT),
            'CB' : self.Volume / (self.lpp * self.beam*DRAFT),
            'CMID' : self.A0,

        }

        self.is_fitted_ = True

        self.result=self.calculate()

        m = self.Volume * self.rho
        B_44 = self.B44_lambda(B_44_hat=self.result.B44HAT, Disp=self.Volume, beam=self.beam, g=self.g, rho=self.rho)
        zeta = self.zeta_lambda(B_1=B_44, GM=self.gm, g=self.g, m=m, omega0=omega0)
        self.parameters={
            'zeta':zeta,
            'omega0':omega0,
        }


    def calculate(self):

        B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT = calculate_roll_damping(**self.ikeda_parameters)
        s = pd.Series()
        s['B44HAT'] = B44HAT
        s['BFHAT'] = BFHAT
        s['BWHAT'] = BWHAT
        s['BEHAT'] = BEHAT
        s['BBKHAT'] = BBKHAT
        return s

