import numpy as np
import pandas as pd

from rolldecayestimators.ikeda import Ikeda
from rolldecayestimators.simplified_ikeda import calculate_roll_damping


class SimplifiedIkeda(Ikeda):
    """
        Class that helps with running various calculations known as the "Simplified Ikedas method" to predict ship roll damping.
        The idea with this class is that it can be subclassed so that varius roll damping contributions can be replaced or
        removed in order to study different "flavors" of the Ikeda methods (I have seen a lot of different implementations).
        """

    def __init__(self, V: np.ndarray, w: np.ndarray, fi_a: float, beam: float, lpp: float,
                 kg: float, volume: float, draught: float, A0: float, lBK=0.0, bBK=0.0,
                 g=9.81, rho=1000.0, visc=1.15 * 10 ** -6, **kwargs):
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
                beam
                    ship beam [m]
                lpp
                    ship perpedicular length [m]
                kg
                    vertical centre of gravity [m] (positive upward)
                volume
                    ship displaced volume [m3]
                draught
                    ship draught [m]
                A0
                    mid section coefficient (A0 = A_mid/(B*T))
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
        self.g = g
        self.w = w
        self.fi_a = fi_a
        self.beam = beam
        self.lpp = lpp
        self.kg = kg
        self.volume = volume
        self.lBK = lBK
        self.bBK = bBK
        self.rho = rho
        self.visc = visc


        N_sections = 21
        x_s = np.linspace(0, lpp, N_sections)
        data = {
            'B_s': beam * np.ones(N_sections),
            'T_s': draught * np.ones(N_sections),
            'C_s': A0 * np.ones(N_sections),
        }
        self.sections = pd.DataFrame(data=data, index=x_s)

        #B_W0: pd.Series

    def calculate_B44(self):
        """
        Calculate total roll damping

        Returns
        -------
        B_44 : ndarray
            Total roll damping [Nm*s/rad]
        """

        B44HAT, BFHAT, BWHAT, BEHAT, BBKHAT, BLHAT = calculate_roll_damping(LPP=self.lpp, CB=self.Cb, CMID=self.A0,
                                                                            OG=self.OG, PHI=np.rad2deg(self.phi_a),
                                                                            lBK=self.lBK, bBK=self.bBK, OMEGA=self.w,
                                                                            DRAFT=self.draught, V=self.V)

        B_44 = (self.calculate_B_W() +
                self.calculate_B_F() +
                self.calculate_B_E() +
                self.calculate_B_L() +
                self.calculate_B_BK()
                )
        return B_44

    def calculate_B_W0(self):
        """
        Calculate roll wave damping at zero speed

        Returns
        -------
        B_W0 : ndarray
            Roll wave damping at zero speed [Nm*s/rad]

        """


    def calculate_B_W(self, Bw_div_Bw0_max=np.inf):
        """
        Calculate roll wave damping at speed

        Returns
        -------
        B_W : ndarray
            Roll wave damping at speed [Nm*s/rad]

        """
        B_W0 = self.calculate_B_W0()
        Bw_div_Bw0 = self.calculate_Bw_div_Bw0()
        B_W = B_W0*Bw_div_Bw0
        return B_W

