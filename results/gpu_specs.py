# gpu_specs.py
# Hardware specifications for plotting reference lines

GPU_SPECS = {
    "NVIDIA_A100": {
        "memory_bandwidth_gbps": 1555,
        "pcie_bandwidth_gbps": 24,
        "name_label": "A100"
    },
    "NVIDIA_L40S": {
        "memory_bandwidth_gbps": 864,
        "pcie_bandwidth_gbps": 24,
        "name_label": "L40S"
    },
    "default": {
        "memory_bandwidth_gbps": 1000,
        "pcie_bandwidth_gbps": 24,
        "name_label": "GPU"
    }
}

def get_gpu_spec(gpu_name):
    """Retrieve GPU specifications for plotting based on the GPU identifier string."""
    if not gpu_name:
        return GPU_SPECS["default"]
        
    # Attempt exact match
    if gpu_name in GPU_SPECS:
        return GPU_SPECS[gpu_name]
    
    # Attempt fuzzy match
    for key, specs in GPU_SPECS.items():
        if key != "default" and key.lower() in gpu_name.lower():
            return specs
            
    return GPU_SPECS["default"]
