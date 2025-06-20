import torch
from main import *

NUM_TASKS = 50  # Define the number of tasks globally

# Run simulation
sim = CoMECSimulator(num_devices=int(NUM_TASKS/1.5), num_tasks=NUM_TASKS, iterations=15000, num_es=4, num_bs=1, algorithm='a0c')
sim.run(residual=True, optimize_for='latency_energy')
tree = sim.iraf_engine.a0c

# Function to extract dataset
def extract_dataset_from_tree(tree, save_path="dataset.pt"):
    dataset = []
    node = tree.root

    while node.children:
        if node.depth == NUM_TASKS - 1:
            print(f"Reached maximum depth {NUM_TASKS - 1} at node {node}.")
            break
        # Get first valid state
        node_state = next((child.state for child in node.children if child.state is not None), None)
        if node_state is None:
            print(f"Node {node} has no valid state in its children, at depth {node.depth}.")
            print("No valid state found in current node's children.")
            break
        for child in node.children:
            if child.action is None:
                continue

            data = {
                'state': torch.tensor(node_state, dtype=torch.float32) if not isinstance(node_state, torch.Tensor) else node_state,
                'action': torch.tensor(child.action, dtype=torch.float32),
                'visit_count': child.N
            }
            dataset.append(data)

        # Traverse to the best visited child
        node = max(node.children, key=lambda x: x.N)

    torch.save(dataset, save_path)
    print(f"✅ Saved dataset with {len(dataset)} entries to {save_path}")

# ✅ Correct function call
extract_dataset_from_tree(tree, "D:/Research/IoT/iRAF-CoMEC-RL/dataset_final_shit.pt")
