# LDP-classifiers
Comparing performance of different classifiers on LDP-perturbed data

Example:
```
from Daan.LDPLogReg import LDPLogReg
from Daan.LDPNaiveBayes import LDPNaiveBayes
from TestSuite import TestSuite

def testRun(database_name=''):
    classifiers = [LDPNaiveBayes(), LDPLogReg(LDPid='DU')]
    testSuite = TestSuite(database_name)
    testSuite.set_params(epsilon_values=[0.1,0.5,1,2,3,5], classifiers=classifiers)
    testSuite.run()

testRun(database_name='wle')```
