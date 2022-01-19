from enum import Enum
import copy
from pure_ldp.frequency_oracles.direct_encoding import DEClient, DEServer
from pure_ldp.frequency_oracles.local_hashing import LHClient, LHServer

class LDPid(Enum):
        DE = 0
        LH = 1

class LDPMechanism:
    def __init__(self, id):
        self.id = id
        self.mechanisms = {LDPid.DE: [DEClient(2,4), DEServer(2,4)], LDPid.LH: [LHClient(2,4, use_olh=True), LHServer(2,4, use_olh=True)]}

    def client(self):
        if self.id in self.mechanisms:
            return copy.copy(self.mechanisms[self.id][0])
        raise IndexError("Specified LDP mechanism not found.")

    def server(self):
        if self.id in self.mechanisms:
            return copy.copy(self.mechanisms[self.id][1])
        raise IndexError("Specified LDP mechanism not found.")