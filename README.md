Below is an example of how to use the TestSuite.

1. Import all the required classifiers
2. Instantiate the required classifiers in the 'classifiers' array. Make sure all the required parameters are also included.
3. Call 'testRun' with the database names you wish to run your test on.
4. If other values for epsilon are required, change these in the 'set_params' call on the testSuite.

```
from Daan.LDPLogReg import LDPLogReg
from Daan.LDPNaiveBayes import LDPNaiveBayes
from TestSuite import TestSuite

def testRun(database_names=[]):
    classifiers = [LDPNaiveBayes(LDPid="DE")]
    testSuite = TestSuite(database_names)
    testSuite.set_params(epsilon_values=[0.01,0.1,0.5,1,2,3,5], classifiers=classifiers, onehotencoded=True)
    testSuite.run()

testRun(database_names=['adult','mushroom','iris','vote','car','nursery'])
```
