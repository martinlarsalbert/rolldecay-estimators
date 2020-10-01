"""
This module contain a class to calculate Ikeda in various ways
"""
import numpy as np
import pandas as pd

from rolldecayestimators import ikeda_speed
from rolldecayestimators import lambdas

class SectionsError(ValueError): pass
class BilgeKeelError(ValueError): pass

import pyscores2.indata
import pyscores2.output

class Ikeda():
    """
    Class that helps with running various calculations known as the "Ikedas method" to predict ship roll damping.
    The idea with this class is that it can be subclassed so that varius roll damping contributions can be replaced or
    removed in order to study different "flavors" of the Ikeda methods (I have seen a lot of different implementations).


    """
    def __init__(self, V:np.ndarray, w:np.ndarray, fi_a:float, B_W0_hat:pd.Series, beam:float, lpp:float,
                 kg:float, volume:float, sections:pd.DataFrame, lBK=0.0, bBK=0.0,
                 g=9.81, rho=1000.0, visc =1.15*10**-6, scale_factor=1.0, **kwargs):
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
        lBK
            length bilge keel [m] (=0 --> no bilge keel)
        bBK
            height bilge keel [m] (=0 --> no bilge keel)
        scale_factor : float
            scale factor is used to calculate the wetted surface in the skin friction damping B_F
        g
            gravity [m/s2]
        rho
            water density [kg/m3]


        Returns
        -------
        None
        """
        self.V = np.array(V, dtype=float)
        self.g = g
        self.w = w
        self.fi_a = fi_a
        self.B_W0_hat=B_W0_hat
        self.beam=beam
        self.lpp=lpp
        self.kg=kg
        self.volume = volume
        self.sections=sections
        self.lBK=lBK
        self.bBK=bBK
        self.rho=rho
        self.visc=visc
        self.scale_factor=scale_factor

    @classmethod
    def load_scoresII(cls, V:np.ndarray, w:np.ndarray, fi_a:float, indata:pyscores2.indata.Indata,
                      output_file:pyscores2.output.OutputFile, lBK: float, bBK :float, g=9.81, rho=1000.0, visc =1.15*10**-6,
                      scale_factor=1.0,
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
        lBK
            bilge keel length [m]
        bBK
            bilge keel height [m]
        g
        rho
        visc
        kwargs
        scale_factor : float
            scale factor is used to calculate the wetted surface in the skin friction damping B_F
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
        kg=(zcg+draught)

        ws, data = output_file.calculate_B_W0()  # Full scale values
        beam_fullscale=beam*scale_factor
        volume_fullscale=volume*scale_factor**3
        ws_hat = lambdas.omega_hat(beam=beam_fullscale, g=g, omega0=ws)
        B_W0_hat_ = lambdas.B_hat_lambda(B=data, Disp=volume_fullscale, beam=beam_fullscale, g=g, rho=rho)
        B_W0_hat = pd.Series(data=B_W0_hat_, index=ws_hat)

        return cls(V=V, w=w, fi_a=fi_a, B_W0_hat=B_W0_hat, beam=beam, lpp=lpp, kg=kg, volume=volume,
                   sections=sections, lBK=lBK, bBK=bBK, g=g, rho=rho, visc=visc, scale_factor=scale_factor)

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
        if hasattr(self,'_R'):
            return self._R
        else:
            raise ValueError('please set the Bilge radius "R"')

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
        #mask = C_s == 0
        #C_s[mask] = 0.000001 * np.max(C_s)  # Putin a very small value
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
        return self.Bw_div_Bw0

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
        return B_W_hat

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

        return B_W0_hat

    def calculate_B_F(self):
        """
        Calculate skin friction damping

        Returns
        -------
        B_F_hat : ndarray
            Skin friction damping [-]

        """

        B_F = ikeda_speed.frictional(w=self.w, fi_a=self.fi_a, V=self.V, B=self.beam, d=self.draught, OG=self.OG,
                                      ra=self.rho, Cb=self.Cb,L=self.lpp, visc=self.visc)
        return self.B_hat(B_F)

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

        B_E_hat=B_E0_hat*(factor)/(1+factor)
        return B_E_hat

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
        return self.B_hat(B_L)

    def calculate_B_BK(self):
        """
        Calculate bilge keel damping

        Returns
        -------
        B_BK_hat : ndarray
            Bilge keel damping [-]

        """

        if np.any(~(self.bBK==0) & (self.lBK==0)):
            raise BilgeKeelError('bBK is 0 but lBK is not!')
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
        return self.B_hat(B44_BK)

class IkedaR(Ikeda):
    """
    Same as Ikeda class but with bilge radius estimation
    """

    @property
    def R(self):
        """
        Bilge radius [m]
        """
        if hasattr(self, '_R'):
            return self._R
        else:
            return self.calculate_R_b()

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



