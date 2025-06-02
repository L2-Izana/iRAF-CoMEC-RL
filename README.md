# iRAF-CoMEC-RL

A Reinforcement Learning-based approach for Intelligent Resource Allocation Framework (iRAF) in Computation Offloading for Mobile Edge Computing (CoMEC) environments.

## Overview

This project implements a Monte Carlo Tree Search (MCTS) based solution for optimizing task scheduling and resource allocation in Mobile Edge Computing environments. The system focuses on minimizing latency and energy consumption while efficiently utilizing available computational resources.

## Features

- MCTS-based task scheduling with UCT exploration
- Progressive Widening mechanism for efficient tree expansion
- Beta sampling as lightweight heuristic guide
- Support for both DNN and non-DNN modes
- Action probability extraction for policy analysis
- Comprehensive performance metrics tracking
- Visualization tools for analysis
- Support for both classical MCTS and AlphaZero-style nodes

## Project Structure

```
.
├── comec_simulator/         # Core simulation components
│   ├── core/               # Base components (BaseStation, EdgeServer, Task)
│   └── visualization/      # Analysis and visualization tools
├── iraf_engine/           # MCTS implementation and scheduling logic
├── result_plots/          # Generated performance plots
└── main.py                # Main execution script
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Additional dependencies listed in requirements.txt

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation:
```bash
python main.py
```

3. View results:
- Check `result_plots/` for generated visualizations
- Analyze metrics in `latency_energy_metrics*.txt` files
- Review action probabilities in `action_probabilities.npy`

## Performance Analysis

The system generates several metrics files:
- `latency_energy_metrics*.txt`: Detailed performance metrics
- `action_probabilities.npy`: Extracted policy information
- `best_action.txt`: Optimal action sequences

## Key Features

### Progressive Widening
- Basic Progressive Widening mechanism in tree expansion phase
- Efficient exploration of action space
- Works seamlessly with both DNN and non-DNN modes

### Beta Sampling
- Lightweight heuristic guide alternative to DNN
- Efficient action selection strategy
- Reduced computational overhead

## Future Improvements

- Enhanced Progressive Widening strategies
- Advanced beta sampling parameters optimization
- Memory optimization for MCTS tree structure
- Parallel tree search implementation
- Enhanced visualization tools

## License

This project is licensed under the terms of the included LICENSE file.
