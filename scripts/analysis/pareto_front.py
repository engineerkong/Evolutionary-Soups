import matplotlib.pyplot as plt
import numpy as np

# Data for each method
data = {
    'SFT': np.array([
        [-0.0316, -0.5753]
    ]),
    'PPO_summary': np.array([
        [0.5542, -0.5727]
    ]),
    'PPO_faithful': np.array([
        [-0.2189, -0.4161]
    ]),
    'RS': np.array([
        [0.5796, -0.5760], [0.4172, -0.5674], [0.2591, -0.5431], [0.1434, -0.5329], [0.0495, -0.5165],
        [-0.0220, -0.4939], [-0.0983, -0.4708], [-0.1546, -0.4505], [-0.1836, -0.4327], [-0.2358, -0.4135]
    ]),
    'MOMoE': np.array([
        [0.0310, -0.4607], [0.0247, -0.4538], [0.0219, -0.4535], [0.0213, -0.4534],
        [0.0186, -0.4533], [0.0174, -0.4520], [0.0172, -0.4500]
    ])
}

# Colors for each method
colors = {
    'SFT': '#FF6B6B',
    'PPO_summary': '#4ECDC4',
    'PPO_faithful': '#FFD93D',
    'RS': '#9B59B6',
    'MOMoE': '#2ECC71'
}

# Markers for each method
markers = {
    'SFT': 'o',
    'PPO_summary': 's',
    'PPO_faithful': '^',
    'RS': 'D',
    'MOMoE': 'p'
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
ref_point = (-4, -4)
ax.scatter(*ref_point, c='#ff6b6b', marker='x', s=150, linewidths=3, zorder=5)
ax.annotate('Reference\n(-4, -4)', ref_point, 
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