import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_smoothed_metrics(files_info, window_size=200, title="Latency + Energy Metrics Comparison"):
    """
    Plot multiple smoothed latency + energy metric curves on the same graph.
    
    Args:
        files_info (list of dict): Each dict should have:
            - 'file_path': path to the metrics file
            - 'label': label for the curve
            - optionally: 'num_devices', 'num_tasks', 'num_bs', 'num_es'
        window_size (int): Window size for simple moving average
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))

    for info in files_info:
        # Load data
        with open(info['file_path'], "r") as f:
            metrics = [float(line.strip()) for line in f if line.strip()]
        metrics_array = np.array(metrics)

        # Compute SMA
        sma = np.convolve(metrics_array, np.ones(window_size) / window_size, mode='valid')
        x = np.arange(window_size - 1, len(metrics_array))

        # Construct label
        label = info.get('label', info['file_path'])
        meta = []
        for k in ['num_devices', 'num_tasks', 'num_bs', 'num_es']:
            if k in info:
                meta.append(f"{info[k]} {k.replace('_', '').upper()}")
        if meta:
            label += f" ({', '.join(meta)})"

        # Plot
        plt.plot(x, sma, label=label, linewidth=2)

    # Formatting
    plt.title(title)
    plt.xlabel("Simulation Step")
    plt.ylabel("Latency + Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_multiple_smoothed_metrics([
    {
        'file_path': "D:/Research/IoT/iRAF-CoMEC-RL/latency_energy_metrics1747671543.9526067.txt",
        'label': 'Case A',
        'num_devices': 10,
        'num_tasks': 20
    },
    {
        'file_path': "D:/Research/IoT/iRAF-CoMEC-RL/latency_energy_metrics1747671860.0425396.txt",
        'label': 'Case B',
        'num_devices': 20,
        'num_tasks': 50
    }
], window_size=200)
