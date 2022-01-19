from LDPMechanism import LDPMechanism, LDPid

class Classifier:
    def __init__(self):
        self.featureData = []
        self.features = []
        self.featureLDPServers = []
        self.classLDPServer = []

    def fit(self, featureData, classData, epsilon, LDPid=LDPid.LH):
        self.LDPMechanism = LDPMechanism(LDPid)
        self.featureData = featureData
        self.classData = classData
        self.epsilon = epsilon
        self.features = featureData.columns