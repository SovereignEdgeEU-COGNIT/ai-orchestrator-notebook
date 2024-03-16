from MLModelInterface import MLModelInterface
import numpy as np
import sys


class InteferenceAwareScheduler(MLModelInterface):

    def get_name(self) -> str:
        return "InteferenceAwareScheduler"

    def initialize(self, models_info):
        # Load models into memory based on provided models_info
        pass

    def predict(self, data):
        # Make a prediction using the current model
        if self.current_model is not None:
            return "Scheduler prediction"
        else:
            return "No model set"

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
