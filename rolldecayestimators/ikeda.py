"""
This module contain a class to calculate Ikeda in various ways
"""
import numpy as np
import pandas as pd

from rolldecayestimators import ikeda_speed

class SectionsError (ValueError): pass


class Ikeda():
    """
    Class that helps with running various calculations known as the "Ikedas method" to predict ship roll damping.
    The idea with this class is that it can be subclassed so that varius roll damping contributions can be replaced or
    removed in order to study different "flavors" of the Ikeda methods (I have seen a lot of different implementations).


    """


    def __init__(self, V:np.ndarray, draught:float, w:np.ndarray, fi_a:float, B_W0:pd.Series, beam:float, lpp:float,
                 kg:float, volume:float, sections:pd.DataFrame, lBK=0.0, bBK=0.0,
                 g=9.81, rho=1000.0, visc =1.15*10**-6, **kwargs):
        """
        Manually specify the inputs to the calculations.
        Note: Some inputs need to be specified here, but others can be defined in other ways
        (by importing a ship geometry etc.)

        Parameters
        ----------
        V
            ship speed [m/s]
        draught
            ship draught [m]
        w
            roll frequency [rad/s]
        fi_a
            roll amplitude [rad]
        B_W0 : pd.Series
            values : wave roll damping for various frequencies [Nm*s/rad]
            index: frequencies [rad/s]
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
        g
            gravity [m/s2]
        rho
            water density [kg/m3]


        Returns
        -------
        None
        """
        self.V = V
        self.draught = draught
        self.g = g
        self.w = w
        self.fi_a = fi_a
        self.B_W0=B_W0
        self.beam=beam
        self.lpp=lpp
        self.kg=kg
        self.volume = volume
        self.sections=sections
        self.lBK=lBK
        self.bBK=bBK
        self.rho=rho
        self.visc=visc

    @property
    def OG(self):
        """
        distance from roll axis (cg) to still water level [m] (positive into water)
        """
        return self.kg - self.draught

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
            ValueError('please set the Bilge radius "R"')

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
        return self.sections['B_s']

    @property
    def T_s(self):
        # Sectional draught [m]
        return self.sections['T_s']

    @property
    def C_s(self):
        # Sectional area coefficient: C_s = S_s/(B_s*T_s)
        return self.sections['C_s']

    @property
    def x_s(self):
        # Sectional x-coordinate from AP [m]
        return self.sections.index

    def verify_sections(self):

        mandatorys=['B_s', 'T_s', 'S_s']
        for mandatory in mandatorys:
            if not hasattr(self,mandatory):
                raise SectionsError('You need to specify sections for this ship')

    def calculate_sectional_lewis_coefficients(self):
        """
        Lewis form approximation' is obtained.
        Given the section's area, S_s, beam B_s and draught T_s, the constants a, a a_3 are uniquely defined
        by von Kerczek and Tuck:
        See code in: ikeda_speed.calculate_sectional_lewis

        Returns
        -------
        a, a_1, a_3 : array_like
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
        B_44 : ndarray
            Total roll damping [Nm*s/rad]
        """
        B_44 = (self.calculate_B_W() +
                self.calculate_B_F() +
                self.calculate_B_E() +
                self.calculate_B_L() +
                self.calculate_B_BK()
                )
        return B_44

    def calculate_B_W(self, Bw_div_Bw0_max=np.inf):
        """
        Calculate roll wave damping at speed

        Returns
        -------
        B_W : ndarray
            Roll wave damping at speed [Nm*s/rad]

        """
        B_W0 = self.calculate_B_W0()
        B_W = ikeda_speed.Bw(w=self.w, V=self.V, d=self.draught, Bw0=B_W0, g=self.g, Bw_div_Bw0_max=Bw_div_Bw0_max)
        return B_W

    def calculate_B_W0(self):
        """
        Calculate roll wave damping at zero speed

        Returns
        -------
        B_W0 : ndarray
            Roll wave damping at zero speed [Nm*s/rad]

        """

        w = self.B_W0.index
        return np.interp(self.w, w, self.B_W0)  # Zero speed wave damping [Nm*s/rad]

    def calculate_B_F(self):
        """
        Calculate skin friction damping

        Returns
        -------
        B_F : ndarray
            Skin friction damping [Nm*s/rad]

        """

        return ikeda_speed.frictional(w=self.w, fi_a=self.fi_a, V=self.V, B=self.beam, d=self.draught, OG=self.OG,
                                      ra=self.rho, Cb=self.Cb,L=self.lpp, visc=self.visc)

    def calculate_B_E(self):
        """
        Calculate bilge eddy damping

        Returns
        -------
        B_E : ndarray
            eddy damping [Nm*s/rad]

        """
        a, a_1, a_3, sigma_s, H = self.calculate_sectional_lewis_coefficients()


        B_E = ikeda_speed.eddy(bwl=self.B_s, a_1=a_1, a_3=a_3, sigma=sigma_s, xs=self.x_s, H0=H, Ts=self.T_s, OG=self.OG,
                               R=self.R, d=self.draught, wE=self.w, fi_a=self.fi_a)

        return B_E

    def calculate_B_L(self):
        """
        Calculate hull lift damping

        Returns
        -------
        B_L : ndarray
            HUl lift damping [Nm*s/rad]

        """
        return ikeda_speed.hull_lift(V=self.V, d=self.draught, OG=self.OG, L=self.lpp, A=self.A_mid, ra=self.rho,
                                     B=self.beam)

    def calculate_B_BK(self):
        """
        Calculate bilge keel damping

        Returns
        -------
        B_BK : ndarray
            Bilge keel damping [Nm*s/rad]

        """
        Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0 = ikeda_speed.bilge_keel(w=self.w, fi_a=self.fi_a, V=self.V, B=self.beam,
                                                                        d=self.draught, A=self.A_mid,
                                      bBK=self.bBK, R=self.R, g=self.g, OG=self.OG, Ho=self.Ho, ra=self.rho)

        B44BK_N0 = Bp44BK_N0*self.lBK
        B44BK_H0 = Bp44BK_H0*self.lBK
        B44BK_L = B44BK_L
        B44_BK = B44BK_N0 + B44BK_H0 + B44BK_L
        return B44_BK

