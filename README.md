from Daan.LDPLogReg import LDPLogReg
from Daan.LDPNaiveBayes import LDPNaiveBayes
from TestSuite import TestSuite

def testRun(database_names=[]):
    classifiers = [LDPNaiveBayes(LDPid="DE")]
    testSuite = TestSuite(database_names)
    testSuite.set_params(epsilon_values=[0.01,0.1,0.5,1,2,3,5], classifiers=classifiers, onehotencoded=True)
    testSuite.run()

testRun(database_names=['adult','mushroom','iris','vote','car','nursery'])
