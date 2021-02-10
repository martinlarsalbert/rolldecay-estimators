"""
This module contain a class to calculate Ikeda in various ways
"""
import numpy as np
import pandas as pd
from numpy import sqrt as sqrt
from scipy.integrate import simps

from rolldecayestimators import ikeda_speed
from rolldecayestimators import ikeda_barge
from rolldecayestimators import simplified_ikeda
from rolldecayestimators import lambdas

class SectionsError(ValueError): pass
class BilgeKeelError(ValueError): pass
class InputError(ValueError): pass
class WInputError(InputError): pass
class PhiAInputError(InputError): pass


import pyscores2.indata
import pyscores2.output

class Ikeda():
    """
    Class that helps with running various calculations known as the "Ikedas method" to predict ship roll damping.
    The idea with this class is that it can be subclassed so that varius roll damping contributions can be replaced or
    removed in order to study different "flavors" of the Ikeda methods (I have seen a lot of different implementations).


    """
    def __init__(self, V:np.ndarray, w:np.ndarray, fi_a:float, B_W0_hat:pd.Series, beam:float, lpp:float,
                 kg:float, volume:float, sections:pd.DataFrame, BKL, BKB, R_b=None,
                 g=9.81, rho=1000.0, visc =1.15*10**-6, scale_factor=1.0, S_f=None, sigma_limit = 0.99, **kwargs):
        """
        Manually specify the inputs to the calculations.
        Note: Some inputs need to be specified here, but others can be defined in other ways
        (by importing a ship geometry etc.)

        Parameters
        ----------
        V
            ship speed [m/s]
        w
            roll frequency [rad/s]
        fi_a
            roll amplitude [rad]
        B_W0_hat : pd.Series
            values : wave roll damping for various frequencies [-]
            index: frequencies hat [-]
        beam
            ship beam [m]
        lpp
            ship perpedicular length [m]
        kg
            vertical centre of gravity [m] (positive upward)
        volume
            ship displaced volume [m3]
        sections
            ship hull geometry sections defined in a pandas data frame:
            sections.index  : section x coordinate measured from AP [m]
            sections['B_s'] : sectional beam [m] (incl. both sides)
            sections['T_s'] : sectional draught [m]
            sections['C_s'] :   sectional area coefficient [-]
                                C_s = S_s/(B_s*T_s)
        BKL
            length bilge keel [m] (=0 --> no bilge keel)
        BKB
            height bilge keel [m] (=0 --> no bilge keel)
        scale_factor : float
            scale factor is used to calculate the wetted surface in the skin friction damping B_F
        g
            gravity [m/s2]
        rho
            water density [kg/m3]
        S_f
            Wetted surface (optional) [m2]
        sigma_limit
            Upper limit to the section area coefficient default: 0.99 (In accordance to Kawahara)

            Kawahara, Y., Maekawa, K., Ikeda, Y.,
            2011. A Simple Prediction Formula of Roll Damping of Conventional Cargo Ships on the Basis of Ikeda’s Method and Its Limitation,
            in: Almeida Santos Neves, M., Belenky, V.L., de Kat, J.O., Spyrou, K., Umeda, N. (Eds.),
            Contemporary Ideas on Ship Stability and Capsizing in Waves, Fluid Mechanics and Its Applications. Springer Netherlands,
            Dordrecht, pp. 465–486. https://doi.org/10.1007/978-94-007-1482-3_26

        R_b float
            Bilge radius for stations [m] (optional)

        Returns
        -------
        None
        """
        self.V = np.array(V, dtype=float)
        self.g = g
        self.w =  w
        self.fi_a =  fi_a
        self.B_W0_hat= B_W0_hat
        self.beam= beam
        self.lpp= lpp
        self.kg= kg
        self.volume =  volume
        self.sections= sections
        self.lBK= BKL
        self.bBK= BKB
        self.rho= rho
        self.visc= visc
        self.scale_factor= scale_factor
        self.S_f = S_f
        self.sigma_limit = sigma_limit

        if not (R_b is None):
            self._R = R_b

    @classmethod
    def load_scoresII(cls, V:np.ndarray, w:np.ndarray, fi_a:float, indata:pyscores2.indata.Indata,
                      output_file:pyscores2.output.OutputFile, BKL: float, BKB :float, kg, g=9.81, rho=1000.0, visc =1.15 * 10 ** -6,
                      scale_factor=1.0, S_f = None, sigma_limit = 0.99, R_b = None,
                      **kwargs):
        """
        Creaate a object from indata and output from ScoresII

        Parameters
        ----------
        V
        w
        fi_a
        indata : pyscores2.indata.Indata
            Indata to ScoresII program
        output_file :pyscores2.output.OutputFile
            Outdata from ScoresII program
        BKL
            bilge keel length [m]
        BKB
            bilge keel height [m]
        kg
            vertical centre of gravity [m] (positive upward)
        g
        rho
        visc
        kwargs
        scale_factor : float
            scale factor is used to calculate the wetted surface in the skin friction damping B_F
        S_f
            Wetted surface (optional) [m2]
        sigma_limit
            Upper limit to the section area coefficient default: 0.99 (In accordance to Kawahara)

            Kawahara, Y., Maekawa, K., Ikeda, Y.,
            2011. A Simple Prediction Formula of Roll Damping of Conventional Cargo Ships on the Basis of Ikeda’s Method and Its Limitation,
            in: Almeida Santos Neves, M., Belenky, V.L., de Kat, J.O., Spyrou, K., Umeda, N. (Eds.),
            Contemporary Ideas on Ship Stability and Capsizing in Waves, Fluid Mechanics and Its Applications. Springer Netherlands,
            Dordrecht, pp. 465–486. https://doi.org/10.1007/978-94-007-1482-3_26

        R_b float
            Bilge radius for stations [m] (optional)

        Returns
        -------

        """

        N_sections = len(indata.bs)
        lpp = indata.lpp/scale_factor
        x_s = np.linspace(0, lpp, N_sections)
        data = {
            'B_s': indata.bs,
            'T_s': indata.ts,
            'C_s': indata.cScores,
        }
        sections = pd.DataFrame(data=data, index=x_s)
        sections['B_s']/=scale_factor
        sections['T_s'] /= scale_factor
        beam=sections['B_s'].max()
        volume=indata.displacement/(scale_factor**3)
        draught=sections['T_s'].max()
        zcg=indata.zcg/scale_factor
        #kg=(zcg+draught)

        ws, data = output_file.calculate_B_W0()  # Full scale values
        beam_fullscale=beam*scale_factor
        volume_fullscale=volume*scale_factor**3
        ws_hat = lambdas.omega_hat(beam=beam_fullscale, g=g, omega0=ws)
        B_W0_hat_ = lambdas.B_hat_lambda(B=data, Disp=volume_fullscale, beam=beam_fullscale, g=g, rho=rho)
        B_W0_hat = pd.Series(data=B_W0_hat_, index=ws_hat)

        return cls(V=V, w=w, fi_a=fi_a, B_W0_hat=B_W0_hat, beam=beam, lpp=lpp, kg=kg, volume=volume,
                   sections=sections, BKL=BKL, BKB=BKB, g=g, rho=rho, visc=visc, scale_factor=scale_factor, S_f=S_f,
                   sigma_limit=sigma_limit, R_b=R_b,
                   **kwargs)

    @property
    def OG(self):
        """
        distance from roll axis (cg) to still water level [m] (positive into water)
        """
        return self.draught - self.kg

    @property
    def Cb(self):
        return self.volume/(self.lpp*self.beam*self.draught)

    @property
    def A_mid(self):
        """
        Mid section area [m2]
        """
        return self.A0*self.beam*self.draught

    @property
    def Ho(self):
        """
        half breadth to draft ratio
        H0 = beam/(2*draught)
        """
        return self.beam/(2*self.draught)

    @property
    def R(self):
        """
        Bilge radius [m]
        """
        if hasattr(self, '_R'):
            return self._R
        else:
            return self.calculate_R_b()

    @R.setter
    def R(self, value):
        self._R = value

    @property
    def S_s(self):
        return self.C_s*self.T_s*self.B_s

    @property
    def A0(self):
        """
        mid section coefficient [-]
        """
        index = int(len(self.C_s)/2)
        return self.C_s[index]

    @property
    def B_s(self):
        # Sectional beam [m] (incl. both sides)
        B_s = np.array(self.sections['B_s'])
        mask=B_s==0
        B_s[mask]=0.000001*np.max(B_s) # Putin a very small value
        return B_s

    @property
    def T_s(self):
        # Sectional draught [m]
        T_s = np.array(self.sections['T_s'])
        mask = T_s == 0
        T_s[mask] = 0.000001*np.max(T_s)  # Putin a very small value
        return T_s

    @property
    def draught(self):
        # draught ship draught [m]
        return np.max(self.T_s)

    @property
    def C_s(self):
        # Sectional area coefficient: C_s = S_s/(B_s*T_s)
        C_s = np.array(self.sections['C_s'])

        ## Apply sigma limit:
        mask = C_s > self.sigma_limit
        C_s[mask] = self.sigma_limit

        return C_s

    @property
    def x_s(self):
        # Sectional x-coordinate from AP [m]
        return np.array(self.sections.index)

    @property
    def w_hat(self):
        return lambdas.omega_hat(beam=self.beam, g=self.g, omega0=self.w)

    @property
    def BD(self):
        return self.beam / self.draught

    @property
    def OGD(self):
        return self.OG / self.draught

    def verify_sections(self):

        mandatorys=['B_s', 'T_s', 'S_s']
        for mandatory in mandatorys:
            if not hasattr(self,mandatory):
                raise SectionsError('You need to specify sections for this ship')

    def B_hat(self,B):
        """
        Nondimensionalize the damping
        Parameters
        ----------
        B
            damping [Nm*rad/s]

        Returns
        -------
        B_hat
            Nondimensional damping coefficient
        """
        return lambdas.B_hat_lambda(B=B, Disp=self.volume, beam=self.beam, g=self.g, rho=self.rho)

    def calculate_sectional_lewis_coefficients(self):
        """
        Lewis form approximation' is obtained.
        Given the section's area, S_s, beam B_s and draught T_s, the constants a, a a_3 are uniquely defined
        by von Kerczek and Tuck:
        See code in: ikeda_speed.calculate_sectional_lewis

        Returns
        -------
        a, a_1, a_3, sigma, H0 : array_like
            sectional lewis coefficients.
        """
        self.verify_sections()
        self.a, self.a_1, self.a_3, self.sigma_s, self.H = ikeda_speed.calculate_sectional_lewis(B_s=self.B_s, T_s=self.T_s, S_s=self.S_s)
        return self.a, self.a_1, self.a_3, self.sigma_s, self.H

    def calculate(self, w=None, fi_a=None) -> pd.DataFrame:
        """
        Calculate all dampings

        Parameters
        ----------
        w : float or ndarray or None (if it has already been defined)
            frequancies (usually natural frequency)
        fi_a : float or ndarray or None (if it has already been defined)
            roll amplitude [rad]

        Returns
        -------
        pd.DataFrame
            DataFrame with ikeda damping
        """

        if w is None:
            if self.w is None:
                raise WInputError('Please specify w')
        else:
            self.w = w

        if fi_a is None:
            if self.fi_a is None:
                raise PhiAInputError('Please specify fi_a')
        else:
            self.fi_a = fi_a

        #if np.isscalar(self.lBK):
        #    if (len(self.fi_a) > 0):
        #        self.lBK *= np.ones(len(self.fi_a))
        #elif len(self.fi_a) != len(self.lBK):
        #        self.lBK = self.lBK[0]*np.ones(len(self.fi_a))
        #
        #if np.isscalar(self.bBK):
        #    if (len(self.fi_a) > 0):
        #        self.bBK *= np.ones(len(self.fi_a))
        #elif len(self.fi_a) != len(self.bBK):
        #        self.bBK = self.bBK[0] * np.ones(len(self.fi_a))

        output = pd.DataFrame()
        output['B_44_hat'] = self._component(output,self.calculate_B44())
        output['B_W0_hat'] = self._component(output,self.calculate_B_W0())
        output['B_W_hat'] = self._component(output,self.calculate_B_W())
        output['B_F_hat'] = self._component(output,self.calculate_B_F())
        output['B_E_hat'] = self._component(output,self.calculate_B_E())
        output['B_BK_hat'] = self._component(output,self.calculate_B_BK())
        output['B_L_hat'] =  self._component(output,self.calculate_B_L())
        output['Bw_div_Bw0'] = self._component(output,self.calculate_Bw_div_Bw0())

        return output

    @staticmethod
    def _component(df,component):

        if len(df) > len(component):
            values = np.tile(component,(len(df),1))
        else:
            values = component

        return values

    def calculate_B44(self):
        """
        Calculate total roll damping

        Returns
        -------
        B_44_hat : ndarray
            Total roll damping [-]
        """
        B_44_hat = (self.calculate_B_W() +
                self.calculate_B_F() +
                self.calculate_B_E() +
                self.calculate_B_L() +
                self.calculate_B_BK()
                )
        return B_44_hat

    def calculate_Bw_div_Bw0(self):
        """
        Calculate Wave damping speed correction

        Returns
        -------
        Bw_div_Bw0
            Bw_div_Bw0 = B_W/B_W0
        """
        self.Bw_div_Bw0 = ikeda_speed.B_W_speed_correction_factor_ikeda(w=self.w, V=self.V, d=self.draught, g=self.g)
        return array(self.Bw_div_Bw0)

    def calculate_B_W(self, Bw_div_Bw0_max=np.inf):
        """
        Calculate roll wave damping at speed

        Returns
        -------
        B_W_hat : ndarray
            Roll wave damping at speed [-]

        """
        B_W0_hat = self.calculate_B_W0()
        Bw_div_Bw0 = self.calculate_Bw_div_Bw0()
        B_W_hat = B_W0_hat*Bw_div_Bw0


        return array(B_W_hat)

    def calculate_B_W0(self):
        """
        Calculate roll wave damping at zero speed

        Returns
        -------
        B_W0_hat : ndarray
            Roll wave damping at zero speed [-]

        """

        w_hat = self.B_W0_hat.index
        B_W0_hat = np.interp(self.w_hat, w_hat, self.B_W0_hat)  # Zero speed wave damping [Nm*s/rad]

        return array(B_W0_hat)

    def calculate_B_F(self):
        """
        Calculate skin friction damping

        Returns
        -------
        B_F_hat : ndarray
            Skin friction damping [-]

        """

        B_F = ikeda_speed.frictional(w=self.w, fi_a=self.fi_a, V=self.V, B=self.beam, d=self.draught, OG=self.OG,
                                      ra=self.rho, Cb=self.Cb,L=self.lpp, visc=self.visc, Sf=self.S_f)
        return array(self.B_hat(B_F))

    def calculate_B_E0(self):
        """
        Calculate bilge eddy damping at zero speed

        Returns
        -------
        B_E0_hat : ndarray
            eddy damping [-]

        """
        a, a_1, a_3, sigma_s, H = self.calculate_sectional_lewis_coefficients()

        B_E0 = ikeda_speed.eddy(bwl=self.B_s, a_1=a_1, a_3=a_3, sigma=sigma_s, xs=self.x_s, H0=H, Ts=self.T_s, OG=self.OG,
                               R=self.R, wE=self.w, fi_a=self.fi_a)
        return array(self.B_hat(B_E0))

    def calculate_B_E(self):
        """
        Calculate bilge eddy damping at speed

        Returns
        -------
        B_E_hat : ndarray
            eddy damping [-]

        """
        B_E0_hat = self.calculate_B_E0()
        #factor=(0.04*self.w*self.lpp/self.V)**2
        factor = np.divide(0.04 * self.w * self.lpp, self.V, out=np.zeros_like(self.V), where=(self.V!=0))**2
        if isinstance(factor,float):
            factor = np.array([factor])

        factor_total = (factor)/(1+factor)
        mask=factor==0
        factor_total[mask]=1.0

        B_E_hat=B_E0_hat*factor_total
        return array(B_E_hat)

    def calculate_B_E0_sections(self):
        """
        Calculate bilge eddy damping for each section at zero speed

        Returns
        -------
        B_E0_s_hat : ndarray
            eddy damping for each sections [-]

        """
        a, a_1, a_3, sigma_s, H = self.calculate_sectional_lewis_coefficients()

        B_E0_s = ikeda_speed.eddy_sections(bwl=self.B_s, a_1=a_1, a_3=a_3, sigma=sigma_s, H0=H, Ts=self.T_s, OG=self.OG,
                               R=self.R, wE=self.w, fi_a=self.fi_a)

        return array(self.B_hat(B_E0_s))

    def calculate_R_b(self):
        """
        Calculate bilge radius with Ikedas empirical formula:
        Returns
        -------
        R_b : ndarray
            Bilge radius [m]

        """
        a, a_1, a_3, sigma_s, H = self.calculate_sectional_lewis_coefficients()

        mask=sigma_s>1
        sigma_s[mask]=0.99  # Do avoid negative value in sqrt
        mask=H<0
        R_b = 2*self.draught*np.sqrt(H*(sigma_s-1)/(np.pi-4))

        mask = (H>=1) & (R_b/self.draught>1)
        R_b[mask]=self.draught

        mask = (H < 1) & (R_b / self.draught > H)
        R_b[mask] = self.beam/2

        return R_b

    def calculate_B_L(self):
        """
        Calculate hull lift damping

        Returns
        -------
        B_L_hat : ndarray
            HUl lift damping [-]

        """
        B_L = ikeda_speed.hull_lift(V=self.V, d=self.draught, OG=self.OG, L=self.lpp, A=self.A_mid, ra=self.rho,
                                     B=self.beam)
        return array(self.B_hat(B_L))

    def calculate_B_BK(self):
        """
        Calculate bilge keel damping

        Returns
        -------
        B_BK_hat : ndarray
            Bilge keel damping [-]

        """

        if np.any(~(self.bBK==0) & (self.lBK==0)):
            raise BilgeKeelError('BKB is 0 but BKL is not!')
            return 0.0

        if isinstance(self.R, np.ndarray):
            index=int(len(self.R)/2)  # Somewhere in the middle of the ship
            R = self.R[index]
        else:
            R = self.R

        Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0 = ikeda_speed.bilge_keel(w=self.w, fi_a=self.fi_a, V=self.V, B=self.beam,
                                                                        d=self.draught, A=self.A_mid,
                                      bBK=self.bBK, R=R, g=self.g, OG=self.OG, Ho=self.Ho, ra=self.rho)

        B44BK_N0 = Bp44BK_N0*self.lBK
        B44BK_H0 = Bp44BK_H0*self.lBK
        B44_BK = B44BK_N0 + B44BK_H0 + B44BK_L

        mask = ((self.lBK == 0) | (pd.isnull(self.lBK)))

        if np.isscalar(mask):
            if mask:
                B44_BK = 0
        else:
            B44_BK[mask] = 0

        return array(self.B_hat(B44_BK))

def array(value):
    if isinstance(value,float):
        return np.array([value])
    else:
        return value

class IkedaR(Ikeda):
    """
    Same as Ikeda class but with mandatory bilge radius
    """

    def __init__(self, V: np.ndarray, w: np.ndarray, fi_a: float, B_W0_hat: pd.Series, beam: float, lpp: float,
                 kg: float, volume: float, sections: pd.DataFrame, BKL, BKB, R_b:float,
                 g=9.81, rho=1000.0, visc=1.15 * 10 ** -6, scale_factor=1.0, S_f=None, sigma_limit = 0.99, **kwargs):

        super().__init__(V=V, w=w, fi_a=fi_a, B_W0_hat=B_W0_hat, beam=beam, lpp=lpp,
                                    kg=kg, volume=volume, sections=sections, BKL=BKL, BKB=BKB,
                                    g=g, rho=rho, visc =visc, scale_factor=scale_factor, S_f=S_f,
                                    sigma_limit=sigma_limit,**kwargs)

        self._R = R_b

    @classmethod
    def load_scoresII(cls, V: np.ndarray, w: np.ndarray, fi_a: float, indata: pyscores2.indata.Indata,
                      output_file: pyscores2.output.OutputFile, BKL: float, BKB: float, kg, R_b:float, g=9.81, rho=1000.0,
                      visc=1.15 * 10 ** -6,
                      scale_factor=1.0, S_f=None, sigma_limit = 0.99,
                      **kwargs):

        estimator = super(cls,cls).load_scoresII(V=V, w=w, fi_a=fi_a, indata=indata, output_file=output_file, BKL=BKL, BKB=BKB, kg=kg, g=g, rho=rho, visc=visc,
                                        scale_factor=scale_factor, R_b=R_b, sigma_limit=sigma_limit, **kwargs)
        return estimator

class IkedaBeSimplified(Ikeda):
    """
       Same as Ikeda class but with eddy damping from Simplified Ikeda
    """

    def calculate_B_E0(self):
        """
        Calculate bilge eddy damping at zero speed

        Returns
        -------
        B_E0_hat : ndarray
            eddy damping [-]

        """
        a, a_1, a_3, sigma_s, H = self.calculate_sectional_lewis_coefficients()

        B_E0 = ikeda_speed.eddy(bwl=self.B_s, a_1=a_1, a_3=a_3, sigma=sigma_s, xs=self.x_s, H0=H, Ts=self.T_s, OG=self.OG,
                               R=self.R, d=self.draught, wE=self.w, fi_a=self.fi_a)

        BD = self.beam/self.draught
        CB = self.Cb
        CMID = self.A0
        OGD = self.OG/self.draught
        OMEGAHAT = self.w_hat
        PHI = np.rad2deg(self.fi_a)  # [deg] !
        B_E0_hat = simplified_ikeda.calculate_B_E(BD, CB, CMID, OGD, OMEGAHAT, PHI)

        return B_E0_hat



class IkedaCarlJohan(Ikeda):
    """
       Same as Ikeda class but with different kind of eddy damping
    """

    def calculate_B_E0(self):
        """
        Calculate bilge eddy damping at zero speed

        Returns
        -------
        B_E0_hat : ndarray
            eddy damping [-]

        """
        a, a_1, a_3, sigma_s, H = self.calculate_sectional_lewis_coefficients()

        B_E0 = ikeda_speed.eddy(bwl=self.B_s, a_1=a_1, a_3=a_3, sigma=sigma_s, xs=self.x_s, H0=H, Ts=self.T_s, OG=self.OG,
                               R=self.R, d=self.draught, wE=self.w, fi_a=self.fi_a)
        return self.B_hat(B_E0)

class IkedaBarge(Ikeda):
    """
       Same as Ikeda class but with barge eddy damping
    """

    def calculate_B_E0(self):
        """
        Calculate bilge eddy damping at zero speed

        Returns
        -------
        B_E0_hat : ndarray
            eddy damping [-]

        """
        B_E0 = ikeda_barge.eddy(rho=self.rho, lpp=self.lpp, d=self.draught, beam=self.beam, OG=self.OG, phi_a=self.fi_a, w=self.w)
        return self.B_hat(B_E0)

class IkedaS(Ikeda):
    """
        Same as Ikeda class but with mandatory wetted surface.
    """

    def __init__(self, V: np.ndarray, w: np.ndarray, fi_a: float, B_W0_hat: pd.Series, beam: float, lpp: float,
                 kg: float, volume: float, sections: pd.DataFrame, BKL, BKB, S_f:float,
                 g=9.81, rho=1000.0, visc=1.15 * 10 ** -6, scale_factor=1.0, **kwargs):

        super().__init__(V=V, w=w, fi_a=fi_a, B_W0_hat=B_W0_hat, beam=beam, lpp=lpp,
                 kg=kg, volume=volume, sections=sections, BKL=BKL, BKB=BKB,
                 g=g, rho=rho, visc =visc, scale_factor=scale_factor, S_f=S_f, **kwargs)

class IkedaCr(Ikeda):
    """
    Same as Ikeda class but with mandatory manual C_r
    """

#    def __init__(self, V: np.ndarray, w: np.ndarray, fi_a: float, B_W0_hat: pd.Series, beam: float, lpp: float,
#                 kg: float, volume: float, sections: pd.DataFrame, BKL, BKB, C_r,
#                 g=9.81, rho=1000.0, visc=1.15 * 10 ** -6, scale_factor=1.0, S_f=None, **kwargs):
#
#        super().__init__(V=V, w=w, fi_a=fi_a, B_W0_hat=B_W0_hat, beam=beam, lpp=lpp,
#                                    kg=kg, volume=volume, sections=sections, BKL=BKL, BKB=BKB,
#                                    g=g, rho=rho, visc =visc, scale_factor=scale_factor, S_f=S_f, **kwargs)
#
#        self.C_r = C_r
#
#    @classmethod
#    def load_scoresII(cls, V: np.ndarray, w: np.ndarray, fi_a: float, indata: pyscores2.indata.Indata,
#                      output_file: pyscores2.output.OutputFile, BKL: float, BKB: float, kg, C_r, g=9.81, rho=1000.0,
#                      visc=1.15 * 10 ** -6,
#                      scale_factor=1.0, S_f=None,
#                      **kwargs):
#
#        estimator = super(cls,cls).load_scoresII(V=V, w=w, fi_a=fi_a, indata=indata, output_file=output_file, BKL=BKL, BKB=BKB, kg=kg, g=g, rho=rho, visc=visc,
#                                        scale_factor=scale_factor, C_r=C_r, **kwargs)
#        return estimator

    def calculate_B_E0(self):
        """
        Calculate bilge eddy damping at zero speed using manually deined C_r

        Returns
        -------
        B_E0_hat : ndarray
            eddy damping [-]

        """
        Bp44E0s = self.eddy_sections()

        Bp44E0 = simps(y=Bp44E0s, x=self.x_s, axis=0)

        return self.B_hat(Bp44E0)

    def eddy_sections(self):

        # WE, CR = np.meshgrid(wE, Cr)
        Cr = np.array(self.C_r)
        wE = np.array(self.w)

        try:
            len_Cr = len(Cr)
        except TypeError:
            len_Cr = 1

        try:
            len_wE = len(wE)
        except TypeError:
            len_wE = 1

        WE = np.tile(wE, (len_Cr, 1))
        FI_a = np.tile(self.fi_a, (len_Cr, 1))

        Tss = np.tile(self.T_s, (len_wE, 1)).transpose()
        CR = np.tile(Cr, (len_wE, 1)).transpose()



        #Bp44E0s = 4 * self.rho * self.T_s**4 * self.w * self.fi_a * self.C_r / (3 * np.pi)

        Bp44E0s = 4 * self.rho * Tss ** 4 * WE * FI_a * CR / (3 * np.pi)

        return Bp44E0s