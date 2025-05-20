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

# plot_multiple_smoothed_metrics([
#     {
#         'file_path': "D:/Research/IoT/iRAF-CoMEC-RL/latency_energy_metrics1747671543.9526067.txt",
#         'label': 'Case A',
#         'num_devices': 10,
#         'num_tasks': 20
#     },
#     {
#         'file_path': "D:/Research/IoT/iRAF-CoMEC-RL/latency_energy_metrics1747671860.0425396.txt",
#         'label': 'Case B',
#         'num_devices': 20,
#         'num_tasks': 50
#     }
# ], window_size=200)


# Load your actual data
# policy = np.load("action_probabilities.npy")


# # Shared state
# current_page = 0
# mode = 'heatmap'  # start in heatmap mode

# # Setup figure and axes
# fig = plt.figure(figsize=(20, 10))
# axs = []

# # === Function: Draw Bar Plots ===
# def draw_page_bars(page):
#     global axs, fig
#     fig.clf()
#     axs = fig.subplots(4, 5, sharey=True).flatten()
#     fig.suptitle(f'Bar Plot Mode — Tasks {page*4} to {page*4+3}', fontsize=16)

#     for i in range(20):  # 4 tasks × 5 heads
#         axs[i].clear()
#         task_idx = page * 4 + i // 5
#         head_idx = i % 5

#         if task_idx < policy.shape[0]:
#             if head_idx == 0:
#                 bins = np.linspace(0, 0.95, 20)
#                 probs = policy[task_idx, head_idx]
#             else:
#                 bins = np.linspace(0, 0.9, 10)
#                 probs = policy[task_idx, head_idx, :10]

#             axs[i].bar(bins, probs, width=(bins[1] - bins[0]) * 0.8)
#             axs[i].set_title(f'T{task_idx}-{head_idx}')
#         else:
#             axs[i].axis('off')

#     fig.canvas.draw_idle()

# # === Function: Draw Heatmap ===
# def draw_page_heatmap(page):
#     global axs, fig
#     fig.clf()
#     axs = fig.subplots(2, 2).flatten()
#     fig.suptitle(f'Heatmap Mode — Tasks {page*4} to {page*4+3}', fontsize=16)

#     for i in range(4):
#         axs[i].clear()
#         task_idx = page * 4 + i

#         if task_idx < policy.shape[0]:
#             matrix = np.zeros((5, 20))
#             for h in range(5):
#                 if h == 0:
#                     matrix[h] = policy[task_idx, h]
#                 else:
#                     matrix[h, :10] = policy[task_idx, h, :10]

#             im = axs[i].imshow(matrix, cmap='plasma', aspect='auto', vmin=0, vmax=1.0)
#             axs[i].set_title(f'Task {task_idx}')
#             axs[i].set_yticks(np.arange(5))
#             axs[i].set_xticks([0, 5, 10, 15, 19])
#             axs[i].set_xlabel('Action bin')
#             axs[i].set_ylabel('Sub-action')
#             fig.colorbar(im, ax=axs[i])
#         else:
#             axs[i].axis('off')

#     fig.canvas.draw_idle()

# # === Key Event Handler ===
# def on_key(event):
#     global current_page, mode
#     if event.key == 'right':
#         if current_page < (policy.shape[0] - 1) // 4:
#             current_page += 1
#     elif event.key == 'left':
#         if current_page > 0:
#             current_page -= 1
#     elif event.key == 'b':
#         mode = 'bar'
#     elif event.key == 'h':
#         mode = 'heatmap'

#     # Redraw based on current mode
#     if mode == 'bar':
#         draw_page_bars(current_page)
#     elif mode == 'heatmap':
#         draw_page_heatmap(current_page)

# import numpy as np
# import matplotlib.pyplot as plt

# # Load your data
# policy = np.load("action_probabilities.npy")  # shape: (20, 5, ...)

# # Setup figure
# fig, axs = plt.subplots(5, 4, figsize=(20, 15))
# fig.suptitle('All 20 Tasks — Minimal Heatmap View (5×4)', fontsize=18)

# for task_idx in range(20):
#     row, col = divmod(task_idx, 4)
#     ax = axs[row, col]

#     # Create the 5x20 matrix for this task
#     matrix = np.zeros((5, 20))
#     for h in range(5):
#         if h == 0:
#             matrix[h] = policy[task_idx, h]
#         else:
#             matrix[h, :10] = policy[task_idx, h, :10]

#     # Plot without any axis elements
#     ax.imshow(matrix, cmap='plasma', aspect='auto', vmin=0, vmax=1.0)
#     ax.set_title(f'Task {task_idx}', fontsize=10)
#     ax.axis('off')  # Hide axes completely

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


plot_multiple_smoothed_metrics([
    {
        'file_path': "D:\Research\IoT\iRAF-CoMEC-RL\latency_energy_metrics1747675621.281468.txt",
        'label': 'Case A',
        'num_devices': 10,
        'num_tasks': 20
    },
])

# plot_action_probabilities()