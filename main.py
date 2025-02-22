import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
data = mnist.data.astype(np.float32)
labels = mnist.target.astype(np.int32)

# Filter for classes 2 and 7
mask = (labels == 2) | (labels == 7)
data = data[mask]
filtered_labels = labels[mask]

# Normalize data (0-1 scale) 
data /= 255.0

# Convert to numpy array and reshape
data_array = np.array(data)
all_images = data_array.reshape(-1, 28, 28)

# Create figure with 28 subplots
plt.figure(figsize=(15, 28))
manager = plt.get_current_fig_manager()
fig, axes = plt.subplots(28, 1, figsize=(15, 28), sharex=True)

# Pre-calculate range array
x_range = np.arange(28)

# Plot all rows at once using vectorized operations, colored by digit
for i in range(28):
    # Plot digit 2 in blue
    mask_2 = (filtered_labels == 2)
    axes[i].plot(x_range, all_images[mask_2, i, :].T, alpha=0.5, color='blue', label='2')
    
    # Plot digit 7 in red
    mask_7 = (filtered_labels == 7)
    axes[i].plot(x_range, all_images[mask_7, i, :].T, alpha=0.5, color='red', label='7')
    
    #axes[i].set_ylabel(f'Row {i+1}')
    axes[i].set_yticks([])
    axes[i].grid(True, linestyle="--", linewidth=0.5)

#plt.xlabel("Pixel Index in Row")
plt.suptitle("Parallel Coordinates Visualization of MNIST Digits 2 (blue) and 7 (red) Row-by-Row")
plt.tight_layout()
plt.savefig('mnist_visualization_2_7.png', dpi=300, bbox_inches='tight')
plt.close()
