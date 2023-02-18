from Daan.LDPLogReg import LDPLogReg
from Daan.LDPNaiveBayes import LDPNaiveBayes
from Membership_Inference.TestSuiteArt import TestSuite

def testRun(db, clas, ep):
    database_names = db
                # Possible values: 
                #   ['adult','iris','mushroom','vote','car','nursery','spect','weightliftingexercises','htru']
    classifiers = clas
                # Possible values: 
                #   LDPNaiveBayes(LDPid="DE"), LDPNaiveBayes(LDPid="UE"),LDPNaiveBayes(LDPid="LH"),LDPNaiveBayes(LDPid="HE"),LDPNaiveBayes(LDPid="HR"),LDPNaiveBayes(LDPid="RAPPOR"), LDPLogReg(LDPid="DU"),LDPLogReg(LDPid="PW"), LDPLogReg(LDPid="HY")
    epsilon_values= ep
                # Possible values: 
                #   [0.01,0.1,0.5,1,2,3,5]
    testSuite = TestSuite(database_names, epsilon_values, classifiers, onehotencoded=False)
    testSuite.run()

testRun(clas=[LDPNaiveBayes(LDPid="RAPPOR"),LDPNaiveBayes(LDPid="DE")],
        db=['adult','iris','mushroom','vote','car', 'nursery','spect','weightliftingexercises','htru'],ep=[0.1,1,5])