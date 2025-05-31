# Physical constants
CHIP_COEFFICIENT = 1e-26
CHANNEL_NOISE_VARIANCE = 1e-5

# Resource capacities
EDGE_SERVER_CPU_CAPACITY = 19.14*1e9  # cycles
BANDWIDTH_PER_BS = 40  # MHz

# Task generation parameters
MIN_DATA_SIZE = 0.2  # MB
MAX_DATA_SIZE = 3.0  # MB
MIN_CPU_CYCLES = 6e9  # cycles
MAX_CPU_CYCLES = 9e10  # cycles
MIN_CHANNEL_GAIN = 0.5
MAX_CHANNEL_GAIN = 1.0
DEFAULT_MAX_LATENCY = 2000  # ms

# Device parameters
DEVICE_CPU_FREQ = 3e8  # Hz
MIN_TRANSMIT_POWER = 32  # mW
MAX_TRANSMIT_POWER = 197  # mW 

# Tree parameters
TREE_STORAGE_BUDGET = 1e7 # 10 million nodes, if more than this, RAM explodes :(
TREE_CONVERGENCE_THRESHOLD = 0.01
TREE_CONVERGENCE_WINDOW = 100

