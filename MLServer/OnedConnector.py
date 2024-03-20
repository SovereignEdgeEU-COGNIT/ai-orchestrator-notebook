import threading
import pyone
import ssl
from os import environ as env


oned_pass = env.get("ONED_PASS")

class OnedConnector:
    def __init__(self):
        if oned_pass is None:
            raise ValueError("ONED_PASS environment variable not set")
        self.one = pyone.OneServer("https://cognit-lab.sovereignedge.eu/RPC2", session=f"oneadmin:{oned_pass}", https_verify=False, timeout=0)
        self.lock = threading.Lock()

    #! Lock setup in this code is bad as it easily leads to deadlocks
    def get_host_green_energy(self, host_id: int) -> float:
        self.lock.acquire()
        host = self.one.host.info(host_id)
        
        if host is None:
            self.lock.release()
            return 0.0
        
        if "ENERGY_RENEWABLE" not in host.TEMPLATE:
            self.lock.release()
            return 0.0
        
        renewable = host.TEMPLATE["ENERGY_RENEWABLE"]
        
        if renewable is None:
            self.lock.release()
            return 0.0
        
        if renewable == "YES":
            self.lock.release()
            return 1.0
        else:
            self.lock.release()
            return 0.0
        
    def get_vm_flavor(self, vm_id: int) -> str:
        """ self.lock.acquire()
        vm = self.one.vm.info(vm_id)
        self.lock.release()
        
        print(vm.USER_TEMPLATE["LOGO"])
        return vm.TEMPLATE["FLAVOR"] """
        
        flavors = ["flavor1", "flavor2", "flavor3", "flavor4", "flavor5"]
        return flavors[vm_id % len(flavors)]
        
        

if __name__ == "__main__":
    oned = OnedConnector()
    print(oned.get_host_green_energy(7))
    print(oned.get_host_green_energy(8))
    print(oned.get_host_green_energy(0))
