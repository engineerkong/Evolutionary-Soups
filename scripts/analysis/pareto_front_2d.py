import matplotlib.pyplot as plt
import numpy as np

# Data for each method
data = {
    'RS': np.array([
        [0.101709, 0.486621], [0.052979, 0.5646], [0.020459, 0.623682], [0.021533, 0.661865], [0.091309, 0.541943],
        [0.558203, -0.24209], [0.729248, -0.50527], [0.804199, -0.65415], [0.869189, -0.76313], [0.910938, -0.81992]
    ]),
    'NEW': np.array([
        [0.011523, 0.587402], [0.025879, 0.470703], [-0.00596, 0.526318], [0.040967, 0.505029], [0.155225, 0.484766],
        [0.384229, 0.328809], [0.653662, 0.023584], [0.818066, -0.31758], [0.89541, -0.62759], [0.92085, -0.77593]
    ]),
    # 'OPT': np.array([
    #     [-0.47954, 1.44367], [0.042566, 0.579073], [0.020559, 0.641855], [0.04367, 0.603508], [0.066781, 0.565162],
    #     [0.540137, -0.23381], [0.67527, -0.44988], [0.810402, -0.66595], [0.851178, -0.73548], [1.269038, -0.73538]
    # ])
}

# Colors for each method
colors = {
    'RS': '#9B59B6',
    'NEW': '#2ECC71',
    # 'OPT': '#1ABC9C'
}

# Markers for each method
markers = {
    'RS': 'D',
    'NEW': 'p',
    # 'OPT': 'h'
}


def get_pareto_front(points):
    """
    Extract the Pareto front from a set of points.
    Assumes maximization of both objectives.
    """
    points = points.copy()
    # Sort by first objective (descending)
    sorted_indices = np.argsort(-points[:, 0])
    sorted_points = points[sorted_indices]
    
    pareto_front = [sorted_points[0]]
    max_y = sorted_points[0, 1]
    
    for point in sorted_points[1:]:
        if point[1] > max_y:
            pareto_front.append(point)
            max_y = point[1]
    
    return np.array(pareto_front)


# Create figure with dark background
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 8))

# Set background colors
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

# Plot each method
for method, points in data.items():
    # Get Pareto front
    pareto = get_pareto_front(points)
    
    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], 
               c=colors[method], marker=markers[method], 
               s=80, label=method, alpha=0.9, edgecolors='white', linewidths=0.5)
    
    # Plot Pareto front line
    ax.plot(pareto[:, 0], pareto[:, 1], 
            c=colors[method], linewidth=2.5, alpha=0.8)

# Plot reference point
ref_point = (-1, -1)
ax.scatter(*ref_point, c='#ff6b6b', marker='x', s=150, linewidths=3, zorder=5)
ax.annotate('Reference\n(-1, -1)', ref_point, 
            textcoords="offset points", xytext=(15, 15),
            fontsize=10, color='#ff6b6b',
            arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1.5))

# Grid
ax.grid(True, linestyle='--', alpha=0.3, color='#3a3a5a')

# Labels and title
ax.set_xlabel('Objective 1', fontsize=14, color='#e0e0e0')
ax.set_ylabel('Objective 2', fontsize=14, color='#e0e0e0')
ax.set_title('Pareto Front Comparison', fontsize=18, fontweight='bold', color='#f0f0f0', pad=20)

# Set axis limits
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Customize ticks
ax.tick_params(colors='#a0a0a0', labelsize=11)

# Legend
legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.9,
                   facecolor='#1a1a2e', edgecolor='#3a3a5a')
for text in legend.get_texts():
    text.set_color('#e0e0e0')

# Adjust layout
plt.tight_layout()

# Save as SVG
plt.savefig('./results/pareto_front_matplotlib.svg', format='svg', 
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)

# Also save as PNG for preview
plt.savefig('./results/pareto_front_matplotlib.png', format='png', 
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)

plt.show()
print("Saved: pareto_front_matplotlib.svg and pareto_front_matplotlib.png")