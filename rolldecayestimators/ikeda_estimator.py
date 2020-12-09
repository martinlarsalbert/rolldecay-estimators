import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from rolldecayestimators.estimator import RollDecay
from rolldecayestimators.direct_estimator import DirectEstimator

from rolldecayestimators.simplified_ikeda import calculate_roll_damping
from rolldecayestimators import equations
from rolldecayestimators import symbols
from rolldecayestimators.substitute_dynamic_symbols import lambdify
from rolldecayestimators.sensitivity import variate_ship, plot_variation, calculate, calculate_variation, _plot_result


from scipy.optimize import curve_fit

class IkedaEstimatorFitError(ValueError): pass

class IkedaEstimator(DirectEstimator):

    eqs = [equations.zeta_equation,  # 0
           equations.omega0_equation_linear]  # 1
    functions_ikeda = [lambdify(sp.solve(eqs, symbols.A_44, symbols.zeta)[0][1]),
                       lambdify(sp.solve(equations.B44_equation, symbols.B_44)[0]),
                       ]

    def __init__(self, lpp:float, TA, TF, beam, BKL, BKB, A0, kg, Volume, gm, V, rho=1000, g=9.81, phi_max=8, omega0=None,
                 verify_input = True, limit_inputs=False, **kwargs):
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
        V
            ship speed [m/s]
        rho
            Density of water [kg/m3]
        g
            acceleration of gravity [m/s**2]
        phi_max
            max roll angle during test [deg]
        omega0
            Natural frequency of motion [rad/s], if None it will be calculated with fft of signal

        For more info see: "rolldecaysestimators/simplified_ikeda.py"
        """
        super().__init__(omega0=omega0)

        self.lpp=lpp
        self.TA=TA
        self.TF=TF
        self.beam=beam
        self.BKL=BKL
        self.BKB=BKB
        self.A0=A0
        self.kg=kg
        self.Volume=Volume
        self.V = V
        self.rho=rho
        self.g=g
        self.gm=gm
        self.phi_max=phi_max
        self.two_point_regression=True
        self.verify_input = verify_input
        self.limit_inputs = limit_inputs

    @property
    def zeta_lambda(self):
        return self.functions_ikeda[0]

    @property
    def B44_lambda(self):
        return self.functions_ikeda[1]

    #def simulate(self, t :np.ndarray, phi0 :float, phi1d0 :float,omega0:float, zeta:float)->pd.DataFrame:
    #    """
    #    Simulate a roll decay test using the quadratic method.
    #    :param t: time vector to be simulated [s]
    #    :param phi0: initial roll angle [rad]
    #    :param phi1d0: initial roll speed [rad/s]
    #    :param omega0: roll natural frequency[rad/s]
    #    :param zeta:linear roll damping [-]
    #    :return: pandas data frame with time series of 'phi' and 'phi1d'
    #    """
    #    parameters={
    #        'omega0':omega0,
    #        'zeta':zeta,
    #    }
    #    return self._simulate(t=t, phi0=phi0, phi1d0=phi1d0, parameters=parameters)

    def fit(self, X, y=None, **kwargs):
        self.X = X

        self.phi_max = np.rad2deg(self.X[self.phi_key].abs().max())  ## Initial roll angle in [deg]

        DRAFT=(self.TA + self.TF) / 2
        omega0=self.omega0

        if (self.lpp*self.beam*DRAFT > 0):
            CB = self.Volume / (self.lpp*self.beam*DRAFT)
        else:
            raise IkedaEstimatorFitError('lpp, beam or DRAFT is zero or nan!')

        self.ikeda_parameters = {

            'LPP' : self.lpp,
            'Beam' : self.beam,
            'DRAFT' : DRAFT,

            'PHI' : self.phi_max,
            'BKL' : self.BKL,
            'BKB' : self.BKB,
            'OMEGA' : omega0,
            'OG' : (-self.kg + DRAFT),
            'CB' : CB,
            'CMID' : self.A0,
            'V':self.V,
            'verify_input':self.verify_input,
            'limit_inputs':self.limit_inputs,

        }

        self.result=self.calculate()

        m = self.Volume * self.rho
        B_44 = self.B44_lambda(B_44_hat=self.result.B_44_HAT, Disp=self.Volume, beam=self.beam, g=self.g, rho=self.rho)
        zeta = self.zeta_lambda(B_1=B_44, GM=self.gm, g=self.g, m=m, omega0=omega0)
        self.parameters={
            'zeta':zeta,
            'omega0':omega0,
            'd':0,
        }

        self.is_fitted_ = True


    def calculate(self,**kwargs):

        B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT, BLHAT = calculate_roll_damping(**self.ikeda_parameters, **kwargs)
        s = pd.Series()
        s['B_44_HAT'] = B44HAT
        s['B_F_HAT'] = BFHAT
        s['B_W_HAT'] = BWHAT
        s['B_E_HAT'] = BEHAT
        s['B_BK_HAT'] = BBKHAT
        s['B_L_HAT'] = BLHAT

        return s

    def result_for_database(self, score=True, **kwargs):

        s = super().result_for_database(score=score, **kwargs)
        s.update(self.result)

        return s

class IkedaQuadraticEstimator(IkedaEstimator):

    functions_ikeda = IkedaEstimator.functions_ikeda
    functions_ikeda.append(lambdify(sp.solve(equations.B_e_equation, symbols.B_e)[0]))       # 2
    functions_ikeda.append(lambdify(sp.solve(equations.zeta_B1_equation, symbols.zeta)[0]))  # 3
    functions_ikeda.append(lambdify(sp.solve(equations.d_B2_equation, symbols.d)[0]))        # 4

    @property
    def B_e_lambda(self):
        return self.functions_ikeda[2]

    @property
    def zeta_B1_lambda(self):
        return self.functions_ikeda[3]

    @property
    def d_B2_lambda(self):
        return self.functions_ikeda[4]

    def fit(self, X=None, y=None, **kwargs):
        self.X = X

        if not self.X is None:
            self.phi_max = np.rad2deg(self.X[self.phi_key].abs().max())  ## Initial roll angle in [deg]


        DRAFT=(self.TA + self.TF) / 2
        omega0=self.omega0

        if (self.lpp*self.beam*DRAFT > 0):
            CB = self.Volume / (self.lpp*self.beam*DRAFT)
        else:
            raise IkedaEstimatorFitError('lpp, beam or DRAFT is zero or nan!')

        self.ikeda_parameters = {

            'LPP' : self.lpp,
            'Beam' : self.beam,
            'DRAFT' : DRAFT,

            'PHI' : self.phi_max,
            'BKL' : self.BKL,
            'BKB' : self.BKB,
            'OMEGA' : omega0,
            'OG' : (-self.kg + DRAFT),
            'CB' : CB,
            'V'  : self.V,
            'CMID' : self.A0,
            'verify_input': self.verify_input,
            'limit_inputs': self.limit_inputs,
        }

        #self.result=self.calculate(**kwargs)
        self.result = {}

        self.result_variation=self.calculate_phi_a_variation()

        if self.two_point_regression:
            B_1_,B_2_ = self.calculate_two_point_regression(**kwargs)
            self.result['B_1'] = B_1 = B_1_['B_44']
            self.result['B_2'] = B_2 = B_2_['B_44']

            B_1_, B_2_ = self.fit_Bs()  # Not used...
        else:
            B_1, B_2 = self.fit_Bs()
            self.result['B_1'] = B_1
            self.result['B_2'] = B_2

        zeta, d = self.Bs_to_zeta_d(B_1=B_1, B_2=B_2)
        factor = 1.0  # Factor
        phi_a = np.abs(np.deg2rad(self.phi_max))/ factor  # [Radians]
        self.result['B_e'] = self.B_e_lambda(B_1=B_1, B_2=B_2, omega0=self.ikeda_parameters['OMEGA'],phi_a=phi_a)

        self.parameters={
            'zeta':zeta,
            'd':d,
            'omega0':omega0,
        }

        self.is_fitted_ = True

    def calculate_two_point_regression(self, **kwargs):
        # Two point regression:
        data = {
            'lpp': self.lpp,
            'beam': self.beam,
            'DRAFT': (self.TA + self.TF) / 2,
            'phi_max': self.phi_max,
            'BKL': self.BKL,
            'BKB': self.BKB,
            'omega0': self.ikeda_parameters['OMEGA'],
            'kg': self.kg,
            'CB': self.ikeda_parameters['CB'],
            'A0': self.A0,
            'V': self.V,
            'Volume': self.Volume,
        }
        row1 = pd.Series(data)
        row1.phi_max *= 0.5
        row2 = pd.Series(data)
        s1_hat = calculate(row1, verify_input=self.verify_input, limit_inputs=self.limit_inputs, **kwargs)
        s2_hat = calculate(row2, verify_input=self.verify_input, limit_inputs=self.limit_inputs, **kwargs)

        s1_hat = self.B44_lambda(B_44_hat=s1_hat, Disp=row1.Volume, beam=row1.beam, g=self.g, rho=self.rho)
        s2_hat = self.B44_lambda(B_44_hat=s2_hat, Disp=row2.Volume, beam=row2.beam, g=self.g, rho=self.rho)

        s1=pd.Series()
        s2=pd.Series()
        for key,value in s1_hat.items():
            new_key = key.replace('_hat','')
            s1[new_key]=s1_hat[key]
            s2[new_key] = s2_hat[key]

        x = np.deg2rad([row1.phi_max, row2.phi_max]) * 8 * row1.omega0 / (3 * np.pi)
        B_2 = (s2 - s1) / (x[1] - x[0])
        B_1 = s1 - B_2 * x[0]

        # Save all of the component as one linear term: _1 and a quadratic term: _2
        for key in s1.index:
            new_name_1 = '%s_1' % key
            self.result[new_name_1] = s1[key]

            new_name_2 = '%s_2' % key
            self.result[new_name_2] = s2[key]

        return B_1,B_2

    def calculate_phi_a_variation(self):

        data={
            'lpp':self.lpp,
            'beam' : self.beam,
            'DRAFT' : (self.TA+self.TF)/2,
            'phi_max' : self.phi_max,
            'BKL' :  self.BKL,
            'BKB':  self.BKB,
            'omega0' : self.ikeda_parameters['OMEGA'],
            'kg' :self.kg,
            'CB':self.ikeda_parameters['CB'],
            'A0' : self.A0,
            'V' : self.V,
            'Volume':self.Volume,
        }
        self.ship = ship = pd.Series(data)

        N = 40
        changes = np.linspace(1, 0.0001, N)
        df_variation = variate_ship(ship=ship, key='phi_max', changes=changes)
        result = calculate_variation(df=df_variation, limit_inputs=self.limit_inputs, verify_input=self.verify_input)
        df_variation['g'] = 9.81
        df_variation['rho'] = 1000
        result = pd.concat((result, df_variation), axis=1)

        result['B_44'] = self.B44_lambda(B_44_hat=result.B_44_hat, Disp=ship.Volume, beam=ship.beam, g=result.g, rho=result.rho)
        result.dropna(inplace=True)
        return result

    def fit_Bs(self):

        def fit(df, B_1, B_2):
            omega0 = df['omega0']
            phi_a = np.deg2rad(df['phi_max'])  # Deg or rad (Radians gave better results actually)???
            #phi_a = df['phi_max']  # Deg or rad???
            return self.B_e_lambda(B_1, B_2, omega0, phi_a)

        coeffs, _ = curve_fit(f=fit, xdata=self.result_variation, ydata=self.result_variation['B_44'])
        B_1 = coeffs[0]
        B_2 = coeffs[1]
        self.result_variation['B_44_fit'] = fit(self.result_variation, *coeffs)
        return B_1,B_2

    def Bs_to_zeta_d(self, B_1, B_2):
        m = self.Volume*self.rho
        zeta = self.zeta_B1_lambda(B_1=B_1, GM=self.gm, g=self.g, m=m, omega0=self.ikeda_parameters['OMEGA'])
        d = self.d_B2_lambda(B_2=B_2, GM=self.gm, g=self.g, m=m, omega0=self.ikeda_parameters['OMEGA'])
        return zeta,d

    def plot_variation(self,ax=None):

        if ax is None:
            fig,ax=plt.subplots()

        self.result_variation.plot(y = ['B_44_hat'], ax=ax)

    def plot_B_fit(self,ax=None):

        if ax is None:
            fig,ax=plt.subplots()

        self.result_variation.plot(y='B_44', ax=ax)
        self.result_variation.plot(y='B_44_fit', ax=ax, style='--')

    @classmethod
    def load(cls, data: {}, X=None):
        """
        Load data and parameters from an existing fitted estimator

        Parameters
        ----------
        data : dict
            Dict containing data for this estimator such as parameters
        X : pd.DataFrame
            DataFrame containing the measurement that this estimator fits (optional).
        Returns
        -------
        estimator
            Loaded with parameters from data and maybe also a loaded measurement X
        """
        estimator = cls(**data)
        estimator.load_data(data=data)
        estimator.load_X(X=X)
        return estimator


