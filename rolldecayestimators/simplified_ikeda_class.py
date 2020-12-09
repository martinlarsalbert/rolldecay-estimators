import numpy as np
import pandas as pd

from rolldecayestimators.ikeda import Ikeda
import rolldecayestimators.ikeda
import rolldecayestimators.simplified_ikeda as si
from rolldecayestimators import ikeda_speed

class SimplifiedIkeda(Ikeda):
    """
        Class that helps with running various calculations known as the "Simplified Ikedas method" to predict ship roll damping.
        The idea with this class is that it can be subclassed so that varius roll damping contributions can be replaced or
        removed in order to study different "flavors" of the Ikeda methods (I have seen a lot of different implementations).
        """

    def __init__(self, V: np.ndarray, w: np.ndarray, fi_a: float, beam: float, lpp: float,
                 kg: float, volume: float, draught: float, A0: float, BKL:float, BKB:float,
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
                BKL
                    length bilge keel [m] (=0 --> no bilge keel)
                BKB
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
        self.lBK = BKL
        self.bBK = BKB
        self.rho = rho
        self.visc = visc

        self._A0=A0
        self._draught=draught



        #B_W0: pd.Series

    @property
    def A0(self):
        return self._A0

    @property
    def draught(self):
        return self._draught

    def calculate_B44(self):
        """
        Calculate total roll damping

        Returns
        -------
        B_44_hat : ndarray
            Nondimensioal total roll damping [-]
        """

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
        B_W0_hat : ndarray
            Nondimensional roll wave damping at zero speed [-]

        """
        B_W0_hat =  si.calculate_B_W0(BD=self.BD, CB=self.Cb, CMID=self.A0, OGD=self.OGD, OMEGAHAT=self.w_hat)
        return B_W0_hat

    def calculate_B_W(self, Bw_div_Bw0_max=np.inf):
        """
        Calculate roll wave damping at speed

        Returns
        -------
        B_W_hat : ndarray
            Nondimensional roll wave damping at speed [-]

        """
        B_W0_hat = self.calculate_B_W0()
        Bw_div_Bw0 = self.calculate_Bw_div_Bw0()
        B_W_hat = B_W0_hat*Bw_div_Bw0
        return B_W_hat

    def calculate_B_F(self):
        """
        Calculate skin friction damping

        Returns
        -------
        B_F_hat : ndarray
            Nondimensional skin friction damping [-]

        """
        B_F_hat = si.calculate_B_F(BD=self.BD, BRTH=self.beam, CB=self.Cb, DRAFT=self.draught, KVC=self.visc,
                                   LPP=self.lpp, OGD=self.OGD, OMEGA=self.w, PHI=np.rad2deg(self.fi_a))
        return B_F_hat

    def calculate_B_E(self):
        """
        Calculate bilge eddy damping

        Returns
        -------
        B_E_hat : ndarray
            Nondimensional eddy damping [-]

        """
        B_E_hat = si.calculate_B_E(BD=self.BD, CB=self.Cb, CMID=self.A0, OGD=self.OGD, OMEGAHAT=self.w_hat,
                                   PHI=np.rad2deg(self.fi_a))
        return B_E_hat

    def calculate_B_BK(self):
        """
        Calculate bilge keel damping

        Returns
        -------
        B_BK_hat : ndarray
            Nondimensional bilge keel damping [-]
        """
        B_BK_hat = si.calculate_B_BK(BBKB=self.bBK/self.beam, BD=self.BD, CB=self.Cb, CMID=self.A0, LBKL=self.lBK/self.lpp, OGD=self.OGD,
                                     OMEGAHAT=self.w_hat, PHI=np.rad2deg(self.fi_a))
        return B_BK_hat


class SimplifiedIkedaBK2(SimplifiedIkeda):

    def calculate_B_BK(self):
        """
        Calculate bilge keel damping

        Returns
        -------
        B_BK_hat : ndarray
            Bilge keel damping [-]

        """

        if np.any(~(self.bBK == 0) & (self.lBK == 0)):
            raise rolldecayestimators.ikeda.BilgeKeelError('BKB is 0 but BKL is not!')
            return 0.0

        if isinstance(self.R, np.ndarray):
            index = int(len(self.R) / 2)  # Somewhere in the middle of the ship
            R = self.R[index]
        else:
            R = self.R

        Bp44BK_N0, Bp44BK_H0, B44BK_L, B44BKW0 = ikeda_speed.bilge_keel(w=self.w, fi_a=self.fi_a, V=self.V, B=self.beam,
                                                                        d=self.draught, A=self.A_mid,
                                                                        bBK=self.bBK, R=R, g=self.g, OG=self.OG,
                                                                        Ho=self.Ho, ra=self.rho)

        B44BK_N0 = Bp44BK_N0 * self.lBK
        B44BK_H0 = Bp44BK_H0 * self.lBK
        B44_BK = B44BK_N0 + B44BK_H0 + B44BK_L

        B44_BK=rolldecayestimators.ikeda.array(self.B_hat(B44_BK))
        mask = ((self.lBK == 0) | (pd.isnull(self.lBK)))
        B44_BK[mask] = 0

        return B44_BK


class SimplifiedIkedaABS(SimplifiedIkeda):

    def calculate_B_W0(self):
        """
        Calculate roll wave damping at zero speed

        Returns
        -------
        B_W0_hat : ndarray
            Nondimensional roll wave damping at zero speed [-]

        """
        return np.abs(super().calculate_B_W0())

    def calculate_B_W(self, Bw_div_Bw0_max=np.inf):
        """
        Calculate roll wave damping at speed

        Returns
        -------
        B_W_hat : ndarray
            Nondimensional roll wave damping at speed [-]

        """
        return np.abs(super().calculate_B_W())

    def calculate_B_F(self):
        """
        Calculate skin friction damping

        Returns
        -------
        B_F_hat : ndarray
            Nondimensional skin friction damping [-]

        """
        return np.abs(super().calculate_B_F())

    def calculate_B_E(self):
        """
        Calculate bilge eddy damping

        Returns
        -------
        B_E_hat : ndarray
            Nondimensional eddy damping [-]

        """
        return np.abs(super().calculate_B_E())

    def calculate_B_BK(self):
        """
        Calculate bilge keel damping

        Returns
        -------
        B_BK_hat : ndarray
            Nondimensional bilge keel damping [-]
        """
        return np.abs(super().calculate_B_BK())