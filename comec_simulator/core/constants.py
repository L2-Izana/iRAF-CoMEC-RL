# Physical constants
CHIP_COEFFICIENT = 1e-26
CHANNEL_NOISE_VARIANCE = 1e-5

# Resource capacities
EDGE_SERVER_CPU_CAPACITY = 19.14*1e9  # cycles
BANDWIDTH_PER_BS = 40  # MHz
NUM_EDGE_SERVERS= 12
NUM_OF_CLUSTERS = 4

# Task generation parameters
MIN_DATA_SIZE = 0.2  # MB
MAX_DATA_SIZE = 3.0  # MB
MIN_CPU_CYCLES = 6e9  # cycles
MAX_CPU_CYCLES = 9e10  # cycles
MIN_CHANNEL_GAIN = 0.5
MAX_CHANNEL_GAIN = 1.0
DEFAULT_MAX_LATENCY = 2000  # ms
ARRIVAL_WINDOW = 10000 # ms

# Device parameters
DEVICE_CPU_FREQ = 3e8  # Hz
MIN_TRANSMIT_POWER = 32  # mW
MAX_TRANSMIT_POWER = 197  # mW 
MIN_NUM_DEVICES = 50
MAX_NUM_DEVICES = 50 # Quite excessive, but we need to test the limits of the simulator

# Tree parameters
TREE_STORAGE_BUDGET = 1e7 # 10 million nodes, if more than this, RAM explodes :(
TREE_CONVERGENCE_THRESHOLD = 0.01
TREE_CONVERGENCE_WINDOW = 100
EXPLORATION_BONUS = 0.8
NUM_SUBACTIONS = 5
NUM_ITERATIONS = 10000

# A0C Algorithm parameters
ADAPTIVE_INITIAL_K = 2
ADAPTIVE_MIN_K = 0.7
ADAPTIVE_INITIAL_ALPHA = 0.8
ADAPTIVE_MIN_ALPHA = 0.3 # Great idea, but not so great results, maybe it's due to the so low count in the end of tree
K_PW = 2.0
ALPHA_PW = 0.7
MAX_PW_FLOOR = 50 # Limit exploration, empirically best results, need further testing and proof 
ALPHA_VAL = 2.0 
BETA_VAL = 0.5

# DNN parameters
DNN_INPUT_DIM = 4 + NUM_EDGE_SERVERS + 1  # 4 for task properties, num_es for edge servers, 1 for base station


