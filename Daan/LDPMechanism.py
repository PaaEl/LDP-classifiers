import copy
import math
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from pure_ldp.core.fo_creator import *


class LDPMechanism:
    def __init__(self, LDPid):
        self.LDPid = LDPid

class LDPFreqMechanism(LDPMechanism):
    def __init__(self, LDPid):
        """ Setup of a pure LDP Frequency Mechanism
        Parameters
        ----------
        LDPid : string 
                Name of the LDPid
        Returns
        -------
        None
        """
        super().__init__(LDPid)

    def client(self, params, hr_server=None):
        """ Create an instance of a client
        Parameters
        ----------
        params : array-like
                 Array of parameters specific for each LDP mechanism
        Returns
        -------
        client : object
                 instance of a pure-ldp client
        """
        if hr_server is not None:
            if self.LDPid in ["RAPPOR"]:
                params = {"f": self._convert_eps_to_f(params["epsilon"]), "m":128, "k":8, "d":params["d"]}
            params['hash_funcs'] = hr_server.get_hash_funcs()
        return create_fo_client_instance(self.LDPid, params)

    def server(self, params):
        """ Create an instance of a server
        Parameters
        ----------
        params : array-like
                 Array of parameters specific for each LDP mechanism
        Returns
        -------
        client : object
                 instance of a pure-ldp server
        """
        if self.LDPid in ["RAPPOR"]:
            f = self._convert_eps_to_f(params["epsilon"])
            f = 0.99 if f >= 1 else f
            params = {"f": f, "m":128, "k":8, "d":params["d"]}
        return create_fo_server_instance(self.LDPid, params)

    def _convert_eps_to_f(self, epsilon):
        """ Convert epsilon into f
        Parameters
        ----------
        epsilon : float
                  The value of the epsilon
        Returns
        -------
        f : float
            The converted value of f
        """
        return round(1/(0.5*math.exp(epsilon/2)+0.5), 2)

class LDPMeanMechanism(LDPMechanism):
    def __init__(self, LDPid):
        """ Setup of a LDP Mean Mechanism
        Parameters
        ----------
        LDPid : string 
                Name of the LDPid
        Returns
        -------
        None
        """
        super().__init__(LDPid)
        self.perturbedData = []

    def perturb(self, t, epsilon):
        """ Perturb the given set of values
        Parameters
        ----------
        t : array-like
            values that need to be perturbed
        epsilon : float
                  value of the float
        Returns
        -------
        perturbed_values : array
                           The perturbed values
        """
        if self.LDPid == "DU":
            return self.perturb_duchi(t, epsilon)
        elif self.LDPid == "PW":
            return self.perturb_piecewise(t, epsilon)
        elif self.LDPid == "HY":
            return self.perturb_piecewise(t, epsilon)

    def perturb_duchi(self, t, epsilon):
        """ Perturb according to the Duchi protocol
        Parameters
        ----------
        t : array-like
            values that need to be perturbed
        epsilon : float
                  value of the float
        Returns
        -------
        perturbed_values : array
                           The perturbed values
        """
        cons = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
        p = ((math.exp(epsilon) - 1) / (2 * (math.exp(epsilon) + 1))) * t + (1/2)
        u = bernoulli.rvs(p, size=p.shape)
        u = u * (2*cons) - cons
        return u

    def perturb_piecewise(self, t, epsilon):
        """ Perturb according to the Piecewise protocol
        Parameters
        ----------
        t : array-like
            values that need to be perturbed
        epsilon : float
                  value of the float
        Returns
        -------
        perturbed_values : array
                           The perturbed values
        """
        expEpsilon = np.exp(epsilon/2)
        tester = expEpsilon / (expEpsilon + 1)
        C = (expEpsilon + 1) / (expEpsilon - 1)
        x = np.random.uniform(size=t.shape)
        l = ((C + 1) / 2) * t - ((C - 1) / 2)
        r = l + C - 1
        # Build a row of 1 and 0 depending on the x < tester
        indicatorPos = np.where(x < tester, 1, 0)
        indicatorNeg = np.ones(t.shape) - indicatorPos
        
        # For the second tester we need a distribution from a split set of data
        testerSlice = np.random.uniform(size=t.shape)
        ratio = (C - abs(l)) / ((C - abs(l) + C - abs(r)))
        indicatorSlice = np.where(testerSlice < ratio, np.random.uniform(-C, l), np.random.uniform(r, C))
        
        return np.random.uniform(l, r, size=t.shape) * indicatorPos + indicatorSlice * indicatorNeg

    def perturb_hybrid(self, t, epsilon):
        """ Perturb according to the Hybrid protocol
        Parameters
        ----------
        t : array-like
            values that need to be perturbed
        epsilon : float
                  value of the float
        Returns
        -------
        perturbed_values : array
                           The perturbed values
        """
        alpha = 0 if epsilon <= 0.61 else (1 - np.exp(-epsilon/2))
        indicatorPos = np.random.binomial(1, alpha, size=t.shape)
        indicatorNeg = np.ones(t.shape) - indicatorPos   
        return self.perturb_piecewise(t) * indicatorPos + self.perturb_duchi(t) * indicatorNeg

    def aggregate(self, perturbedData):
        """ Aggregate the perturbed data
        Parameters
        ----------
        perturbedData : array-like
                        The perturbed values
        Returns
        -------
        mean_value : float
                     The mean taken from the aggregated values.
        """
        n = len(perturbedData)
        return (1/n * np.sum(perturbedData, axis=0))