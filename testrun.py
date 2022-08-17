from Daan.LDPLogReg import LDPLogReg
from Daan.LDPNaiveBayes import LDPNaiveBayes
from TestSuite import TestSuite

def testRun():
    database_names = ['car'] 
                # Possible values: 
                #   ['adult','iris','mushroom','vote','car','nursery','spect','weightliftingexercises','htru']
    classifiers = [LDPNaiveBayes(LDPid="DE")]
                # Possible values: 
                #   LDPNaiveBayes(LDPid="DE"), LDPNaiveBayes(LDPid="UE"),LDPNaiveBayes(LDPid="LH"),LDPNaiveBayes(LDPid="HE"),LDPNaiveBayes(LDPid="HR"),LDPNaiveBayes(LDPid="RAPPOR"), LDPLogReg(LDPid="DU"),LDPLogReg(LDPid="PW"), LDPLogReg(LDPid="HY")
    epsilon_values= [1]
                # Possible values: 
                #   [0.01,0.1,0.5,1,2,3,5]
    testSuite = TestSuite(database_names, epsilon_values, classifiers, onehotencoded=False)
    testSuite.run()

testRun()