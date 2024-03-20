import threading
from typing import Dict, List
from ClassifierInterface import ClassifierInterface
import numpy as np
import sys
from DBConnector import DBClient, Host, Vm, Metric
from SchedulerInterface import SchedulerInterface
from OnedConnector import OnedConnector


class InteferenceAwareScheduler(SchedulerInterface):
    
    def __init__(self, db: DBClient, oned: OnedConnector, classifier: ClassifierInterface) -> None:
        self.db = db
        self.oned = oned
         # 2D array of host interference-aware values
        self.ia_width = classifier.get_output_size()
        self.hosts_ia_vals = np.ndarray((0, self.ia_width))
        self.hosts_green_energy = np.ndarray((0, 1))
        self.hosts_to_index_map: Dict[str, int] = {}
        self.index_to_hosts_map: Dict[int, str] = {}
        #attributes = 6 # cpu, mem, disk_read, disk_write, net_rx, net_tx
        self.attribute_weights = np.ones(self.ia_width) / self.ia_width
        self.classifier = classifier
        self.green_energy_scalar = 1
        self.lock = threading.Lock()
     
    @staticmethod
    def get_name() -> str:
        return "InteferenceAwareScheduler"
    
    def cleanup(self): #! Rename
        self.db.close_all_connections()
        
    def set_green_energy_scalar(self, green_energy_scalar: int):
        self.green_energy_scalar = green_energy_scalar
        
    def set_classifier(self, classifier: ClassifierInterface):
        self.classifier = classifier
        self.ia_width = classifier.get_output_size()
        self.hosts_ia_vals = np.ndarray((0, self.ia_width))
        self.attribute_weights = np.ones(self.ia_width) / self.ia_width
        self.initialize()

    def initialize(self) -> None:
        hosts = self.db.fetch_hosts()
            
        self.lock.acquire()
        
        for host in hosts:
            
            #if host.usage_cpu == 0:
                #continue
            host_init_ia_vals = np.zeros(self.ia_width)
            host_green_energy = np.random.rand(1) #! This should be the actual green energy value
            #host_green_energy = self.oned.get_host_green_energy(int(host.hostid))
            #print("host_green_energy\n", host_green_energy)
            
            self.hosts_ia_vals = np.vstack([self.hosts_ia_vals, host_init_ia_vals])
            self.hosts_green_energy = np.vstack([self.hosts_green_energy, host_green_energy])
            self.hosts_to_index_map[host.hostid] = len(self.hosts_ia_vals) - 1
            self.index_to_hosts_map[len(self.hosts_ia_vals) - 1] = host.hostid
            print(host.hostid)
            
            
        vms = self.db.fetch_vms()
        
        for vm in vms:
            if vm.hostid is None or vm.hostid == '' or vm.vmid is None or vm.vmid == '':
                continue
            #! I think the ML algorithm will give me vals between 0 and 1 so mult by 100
            vm_ia_vals = self.classifier.predict(int(vm.vmid))
            #print("vm_ia_vals\n", vm_ia_vals)
            vm_ia_vals = np.multiply(vm_ia_vals, 100)
            #print("vm_ia_vals after mult by 100\n", vm_ia_vals)
            
            host_index = self.hosts_to_index_map[vm.hostid]
            self.hosts_ia_vals[host_index] += vm_ia_vals
            
        self.lock.release()

    def predict(self, vm_id: int, host_ids: List[int]) -> int:
        
        vms = self.db.fetch_vm(vm_id)
        
        #! I think the ML algorithm will give me vals between 0 and 1 so mult by 100
        
        vm_ia_vals = self.classifier.predict(vm_id)
        print("vm_ia_vals\n", vm_ia_vals)
        vm_ia_vals = np.multiply(vm_ia_vals, 100)
        print("vm_ia_vals after mult by 100\n", vm_ia_vals)
        
        # The indicies and rev map are confusing, need to clean this up
        host_indicies = []
        host_subset_indicies_index_map = {}
        for host_id in host_ids:
            host_index = self.hosts_to_index_map.get(str(host_id))
            if host_index is not None:
                host_indicies.append(host_index)
                host_subset_indicies_index_map[len(host_indicies) - 1] = host_index
                
        hosts_ia_vals_subset = self.hosts_ia_vals[np.array(host_indicies)]
        hosts_green_energy_subset = self.hosts_green_energy[np.array(host_indicies)]
        
        self.lock.acquire()
        
        host_subset_index = self.schedule_vm(vm_ia_vals, hosts_ia_vals_subset, hosts_green_energy_subset)
        #print(host_subset_index)
        host_index = host_subset_indicies_index_map[host_subset_index]
        self.hosts_ia_vals[host_index] += vm_ia_vals
        
        if len(vms) == 1 and vms[0] is not None:
            vm = vms[0]
            if vm.hostid is not None and vm.hostid != '':
                print("works")
                prev_host_index = self.hosts_to_index_map[vm.hostid]
                self.hosts_ia_vals[prev_host_index] -= vm_ia_vals
        
        self.lock.release()
        
        host_id = self.index_to_hosts_map.get(int(host_index))
        
        return int(host_id or -1)
    
    
        
    def schedule_vm(self, vm_ia_vals, host_ia_vals, hosts_green_energy) -> np.intp:
        # Adjust VM attributes based on weights
        weighted_vm_ia_vals = vm_ia_vals * self.attribute_weights
        
        # Calculate weighted host attributes
        weighted_host_attributes = host_ia_vals * self.attribute_weights
        print("weighted_host_attributes\n", weighted_host_attributes)
        
        # Calculate potential new load for each host after adding the VM, normalized by attribute importance
        potential_new_load = weighted_host_attributes + weighted_vm_ia_vals
        print("potential_new_load\n", potential_new_load)
        
        # Calculate the load from the mean for each host
        #host_load_from_mean = host_loads - np.mean(host_loads)
        #print("host_load_from_mean\n", host_load_from_mean)

        # Apply the weight to the potential new load
        #potential_new_load += potential_new_load * (host_load_from_mean[:, np.newaxis]) * host_loading_scalar
        #print("potential_new_load after loading scaling\n", potential_new_load)
        
        # Apply lack of green energy cost
        potential_new_load += potential_new_load * (1 - hosts_green_energy) * self.green_energy_scalar
        print("potential_new_load after green energy cost\n", potential_new_load)

        # Lower scores are better. This is a simplified scoring mechanism; adjust weights as needed.
        scores = np.sum(potential_new_load, axis=1) #- green_energy * green_energy_weight + host_loads * host_load_weight  # Weight green energy and load heavily
        print("scores\n", scores)
        
        # Choose the host with the lowest score
        chosen_host_idx = np.argmin(scores)
        
        return chosen_host_idx

"""
def calculate_host_load(host_attributes, host_max_attributes, attribute_weights):
    # Calculate host load per attribute
    host_loads = np.sum((host_attributes / host_max_attributes) * attribute_weights, axis=1) 
    return host_loads

def schedule_vm(vm_attributes, host_attributes, attribute_weights, green_energy, host_loads, green_energy_weight=100, host_loading_scalar=1):
    # Adjust VM attributes based on weights
    weighted_vm_attributes = vm_attributes * attribute_weights
    
    # Calculate weighted host attributes
    weighted_host_attributes = host_attributes * attribute_weights
    print("weighted_host_attributes\n", weighted_host_attributes)
    
    # Calculate potential new load for each host after adding the VM, normalized by attribute importance
    potential_new_load = weighted_host_attributes + weighted_vm_attributes
    print("potential_new_load\n", potential_new_load)
    
    # Calculate the load from the mean for each host
    host_load_from_mean = host_loads - np.mean(host_loads)
    print("host_load_from_mean\n", host_load_from_mean)

    # Apply the weight to the potential new load
    potential_new_load += potential_new_load * (host_load_from_mean[:, np.newaxis]) * host_loading_scalar
    print("potential_new_load after loading scaling\n", potential_new_load)

    # Lower scores are better. This is a simplified scoring mechanism; adjust weights as needed.
    scores = np.sum(potential_new_load, axis=1) #- green_energy * green_energy_weight + host_loads * host_load_weight  # Weight green energy and load heavily
    print("scores\n", scores)
    
    # Choose the host with the lowest score
    chosen_host_idx = np.argmin(scores)
    
    return chosen_host_idx

    
if __name__ == "__main__":
        
    np.random.seed(0)


    # Example number of hosts
    hosts = 3
    attrbutes = 5

    # Generate the attributes weights between 0.1 and 0.35 and normalize them to sum to 1
    attribute_weights = np.random.rand(attrbutes) * 0.25 + 0.1
    attribute_weights /= np.sum(attribute_weights)

    # get cli arg "-eq", if exists then all attributes have same weight
    if len(sys.argv) > 1 and sys.argv[1] == "-eq":
        attribute_weights = np.ones(attrbutes) / attrbutes

    print("attribute_weights\n", attribute_weights)

    # Example hosts with their current total attributes (randomly generated for 10 hosts, N=5 attributes each)
    host_max_attributes = np.random.rand(hosts, attrbutes) * 200 + 100 # Values between 40-240 for simplicity
    print("host_max_attributes\n", host_max_attributes)

    host_attributes = np.zeros((hosts, attrbutes))

    # Example VM to be scheduled, with its attributes
    #vm_attributes = np.array([20, 15, 30, 25, 10])

    def generate_vm():
        return np.random.rand(attrbutes) * 20  # Randomly generate VM attributes between 0-20

    # Example green energy metrics for hosts (0-1, where 1 is 100% green)
    #green_energy = np.array([0.9, 0.7, 0.8])
    green_energy = np.random.rand(hosts)
    print("random_green_energy\n", green_energy)

    
    # While loop that waits for user input to schedule a VM
    while True:
        print("Press enter to schedule a VM")
        input()
        
        # Schedule the VM
        vm_attributes = generate_vm()
        print(f"VM attributes:\n {vm_attributes}")
        host_loads = calculate_host_load(host_attributes, host_max_attributes, attribute_weights)
        print(f"host_loads\n {host_loads}")

        chosen_host_idx = schedule_vm(vm_attributes, host_attributes, attribute_weights, green_energy, host_loads, 0, 1)
        host_attributes[chosen_host_idx] += vm_attributes  # Update host attributes

        print(f"VM should be scheduled on host: {chosen_host_idx}")
"""