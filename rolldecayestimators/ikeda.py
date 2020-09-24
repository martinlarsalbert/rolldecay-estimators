"""
This module contain a class to calculate Ikeda in various ways
"""

from rolldecayestimators import ikeda_speed
from collections import OrderedDict

class SectionsError (ValueError): pass


class Ikeda():
    """
    Class that helps with running various calculations known as the "Ikedas method" to predict ship roll damping.
    The idea with this class is that it can be subclassed so that varius roll damping contributions can be replaced or
    removed.


    """

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

    def calculate_B_W(self):
        """
        Calculate roll wave damping

        Returns
        -------
        B_W : ndarray
            Roll wave damping [Nm*s/rad]

        """
        return 0.0

    def calculate_B_F(self):
        """
        Calculate skin friction damping

        Returns
        -------
        B_F : ndarray
            Skin friction damping [Nm*s/rad]

        """
        return 0.0

    def calculate_B_E(self):
        """
        Calculate bilge eddy damping

        Returns
        -------
        B_E : ndarray
            eddy damping [Nm*s/rad]

        """
        return 0.0

    def calculate_B_L(self):
        """
        Calculate hull lift damping

        Returns
        -------
        B_L : ndarray
            HUl lift damping [Nm*s/rad]

        """
        return 0.0

    def calculate_B_BK(self):
        """
        Calculate bilge keel damping

        Returns
        -------
        B_BK : ndarray
            Bilge keel damping [Nm*s/rad]

        """
        return 0.0

