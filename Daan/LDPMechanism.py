import copy
import math
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from pure_ldp.frequency_oracles.direct_encoding import DEClient, DEServer
from pure_ldp.frequency_oracles.local_hashing import LHClient, LHServer
from pure_ldp.frequency_oracles.rappor import RAPPORClient, RAPPORServer


class LDPMechanism:
    def __init__(self, LDPid):
        self.LDPid = LDPid

class LDPFreqMechanism(LDPMechanism):
    def __init__(self, LDPid):
        super().__init__(LDPid)
        self.mechanisms = {'DE': [DEClient(2,4), DEServer(2,4)], 
            'LH': [LHClient(2,4, use_olh=True), LHServer(2,4, use_olh=True)]}

    def client(self):
        if self.LDPid in self.mechanisms:
            return copy.copy(self.mechanisms[self.LDPid][0])
        raise IndexError("Specified LDP mechanism not found.")

    def server(self):
        if self.LDPid in self.mechanisms:
            return copy.copy(self.mechanisms[self.LDPid][1])
        raise IndexError("Specified LDP mechanism not found.")

class LDPMeanMechanism(LDPMechanism):
    def __init__(self, LDPid):
        super().__init__(LDPid)
        self.perturbedData = []

    def perturb(self, t, epsilon):
        if self.LDPid == "DU":
            return self.perturb_duchi(t, epsilon)
        elif self.LDPid == "PW":
            return pd.DataFrame(t).applymap(lambda x: self.perturb_piecewise(x, epsilon))
        elif self.LDPid == "HY":
            return pd.DataFrame(t).applymap(lambda x: self.perturb_hybrid(x, epsilon))

        # cons = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
        # p = ((math.exp(epsilon) - 1) / (2 * (math.exp(epsilon) + 1))) * t + (1/2)
        # u = bernoulli.rvs(p, size=p.shape)
        # u = u * (2*cons) - cons
        # return u

        # returnArray = []
        # expEpsilon = np.exp(epsilon/2)
        # tester = expEpsilon / (expEpsilon + 1)
        # C = (expEpsilon + 1) / (expEpsilon - 1)
        # for feat in t:
        #     featArray = []
        #     for ti in feat:
        #         x = np.random.uniform()
        #         l = ((C + 1) / 2) * ti - ((C - 1) / 2)
        #         r = l + C - 1
        #         if x < tester:
        #             featArray.append(np.random.uniform(l, r))
        #         else:
        #             temp = np.random.uniform(-C, C)
        #             while (temp < r and temp > l):
        #                 temp = np.random.uniform(-C, C)
        #             featArray.append(temp)
        #     returnArray.append(featArray)
        # return returnArray

    def perturb_duchi(self, t, epsilon):
        cons = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
        p = ((math.exp(epsilon) - 1) / (2 * (math.exp(epsilon) + 1))) * t + (1/2)
        u = bernoulli.rvs(p, size=p.shape)
        u = u * (2*cons) - cons
        return u

    def perturb_piecewise(self, t, epsilon):
        # t is a single value

        expEpsilon = np.exp(epsilon/2)
        tester = expEpsilon / (expEpsilon + 1)
        C = (expEpsilon + 1) / (expEpsilon - 1)
        x = np.random.uniform()
        l = ((C + 1) / 2) * t - ((C - 1) / 2)
        r = l + C - 1
        if x < tester:
            return np.random.uniform(l, r)
        else:
            temp = np.random.uniform(-C, C)
            while (temp < r and temp > l):
                temp = np.random.uniform(-C, C)
            return temp

        # returnArray = []
        # expEpsilon = np.exp(epsilon/2)
        # tester = expEpsilon / (expEpsilon + 1)
        # C = (expEpsilon + 1) / (expEpsilon - 1)
        # for feat in t:
        #     featArray = []
        #     for ti in feat:
        #         x = np.random.uniform()
        #         l = ((C + 1) / 2) * ti - ((C - 1) / 2)
        #         r = l + C - 1
        #         if x < tester:
        #             featArray.append(np.random.uniform(l, r))
        #         else:
        #             temp = np.random.uniform(-C, C)
        #             while (temp < r and temp > l):
        #                 temp = np.random.uniform(-C, C)
        #             featArray.append(temp)
        #     returnArray.append(featArray)
        # return returnArray

    def perturb_hybrid(self, t, epsilon):
        alpha = 0 if epsilon <= 0.61 else (1 - np.exp(-epsilon/2))
        if np.random.binomial(1, alpha):
            return self.perturb_piecewise(t)
        else:
            return self.perturb_duchi(np.array(t))
        
        # perturbedData = []
        # for ti in t:
        #     if np.random.binomial(1, alpha):
        #         perturbedData = perturbedData + self.perturb_piecewise(np.array([ti]))
        #     else:
        #         perturbedData = perturbedData + [self.perturb_duchi(np.array([ti]))]
        # return perturbedData
        

    def aggregate(self, perturbedData):
        n = len(perturbedData)
        return (1/n * np.sum(perturbedData, axis=0))