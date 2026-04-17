import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Data from your log
data = {
    "expert0": [0.1123, -0.4231, -1.8223],
    "expert1": [-0.9967, -0.3984, -1.7506],
    "expert2": [-1.0419, -0.4698, 1.9390],
    "MoE fixed [0,0,1]": [-1.0419, -0.4698, 1.9390],
    "MoE fixed [0,0.2,0.8]": [-1.0585, -0.4821, 1.8188],
    "MoE fixed [0,0.4,0.6]": [-0.8504, -0.4224, -0.9769],
    "MoE fixed [0,0.6,0.4]": [-0.9585, -0.3865, -1.1752],
    "MoE fixed [0,0.8,0.2]": [-0.9925, -0.3951, -1.6435],
    "MoE fixed [0,1,0]": [-0.9967, -0.3984, -1.7506],
    "MoE fixed [0.2,0,0.8]": [-1.0065, -0.5096, 1.7979],
    "MoE fixed [0.2,0.2,0.6]": [-0.7531, -0.4532, -0.7682],
    "MoE fixed [0.2,0.4,0.4]": [-0.8542, -0.4303, -1.1902],
    "MoE fixed [0.2,0.6,0.2]": [-0.9640, -0.3933, -1.5629],
    "MoE fixed [0.2,0.8,0]": [-0.8672, -0.3683, -1.8504],
    "MoE fixed [0.4,0,0.6]": [-1.2299, -0.4788, 1.3264],
    "MoE fixed [0.4,0.2,0.4]": [-0.7162, -0.4300, -1.5139],
    "MoE fixed [0.4,0.4,0.2]": [-1.0732, -0.4835, -2.1875],
    "MoE fixed [0.4,0.6,0]": [-0.8094, -0.3910, -1.9669],
    "MoE fixed [0.6,0,0.4]": [-0.6681, -0.4730, -1.0694],
    "MoE fixed [0.6,0.2,0.2]": [-0.6415, -0.4293, -1.3063],
    "MoE fixed [0.6,0.4,0]": [-1.2468, -0.4346, -1.2389],
    "MoE fixed [0.8,0,0.2]": [-0.5312, -0.4753, -1.4294],
    "MoE fixed [0.8,0.2,0]": [-0.5985, -0.4500, -1.5502],
    "MoE fixed [1,0,0]": [0.1123, -0.4231, -1.8223],
    "MoE NSGAII ind_000": [-0.4261, -0.4545, -1.9595],
    "MoE NSGAII ind_001": [-1.0604, -0.5098, 1.8002],
    "MoE NSGAII ind_002": [-0.9085, -0.3503, -1.6742],
    "MoE NSGAII ind_003": [-1.0269, -0.5182, 1.9423],
    "MoE NSGAII ind_004": [-1.0188, -0.5054, 1.5057],
    "MoE NSGAII ind_005": [-0.7846, -0.4554, -0.3642],
    "MoE NSGAII ind_006": [-0.7224, -0.3654, -1.0226],
    "MoE NSGAII ind_007": [-0.6600, -0.4091, -1.2928],
    "MoE NSGAII ind_008": [-0.6677, -0.4294, -1.2361],
    "MoE NSGAII ind_009": [-0.5541, -0.4433, -1.6188],
    "MoE NSGAII ind_010": [-1.0313, -0.4787, 1.5461],
    "MoE NSGAII ind_011": [-0.7694, -0.4648, -0.5553],
    "MoE NSGAII ind_012": [-0.4995, -0.4532, -1.7922],
    "MoE NSGAII ind_013": [-0.6934, -0.4437, -0.8394],
    "MoE NSGAII ind_014": [-1.0389, -0.4973, 1.7675],
    "MoE NSGAII ind_015": [-0.6097, -0.4581, -1.0657],
    "MoE NSGAII ind_016": [-0.7404, -0.4380, -0.6441],
    "MoE NSGAII ind_017": [-0.6463, -0.4301, -1.3467],
    "MoE NSGAII ind_018": [-0.6509, -0.4698, -0.9247],
    "MoE NSGAII ind_019": [-0.4944, -0.4574, -1.6481],
    "MoE NSGAIIv2 ind_000": [-0.4419, -0.4556, -1.8144],
    "MoE NSGAIIv2 ind_001": [-1.0702, -0.5143, 1.7179],
    "MoE NSGAIIv2 ind_002": [-1.1053, -0.3856, -2.0880],
    "MoE NSGAIIv2 ind_003": [-1.0455, -0.5023, 1.7123],
    "MoE NSGAIIv2 ind_004": [-0.7207, -0.4265, -1.4216],
    "MoE NSGAIIv2 ind_005": [-0.7514, -0.4795, -1.3024],
    "MoE NSGAIIv2 ind_006": [-1.0479, -0.5061, 1.3793],
    "MoE NSGAIIv2 ind_007": [-0.7098, -0.4340, -1.5107],
    "MoE NSGAIIv2 ind_008": [-0.4450, -0.4638, -1.5340],
    "MoE NSGAIIv2 ind_009": [-1.0607, -0.4064, -1.6102],
    "MoE NSGAIIv2 ind_010": [-0.4624, -0.4561, -1.8208],
    "MoE NSGAIIv2 ind_011": [-0.9842, -0.4538, -2.2050],
    "MoE NSGAIIv2 ind_012": [-0.8719, -0.4610, -2.6450],
    "MoE NSGAIIv2 ind_013": [-0.6701, -0.4286, -0.8614],
    "MoE NSGAIIv2 ind_014": [-0.6796, -0.4293, -1.3754],
    "MoE NSGAIIv2 ind_015": [-0.9954, -0.4438, -2.1904],
    "MoE NSGAIIv2 ind_016": [-0.7439, -0.4625, -0.6974],
    "MoE NSGAIIv2 ind_017": [-0.9622, -0.4137, -1.1973],
    "MoE NSGAIIv2 ind_018": [-0.9631, -0.4116, -1.5736],
    "MoE NSGAIIv2 ind_019": [-1.1860, -0.4219, -1.1188],
}

# Create output directory with timestamp
output_dir = f"results/analysis/pareto_figures_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving figures to: {output_dir}/")

# Group data
expert_data = {k: v for k, v in data.items() if "expert" in k}
fixed_data = {k: v for k, v in data.items() if "MoE fixed" in k}
nsgaii_data = {k: v for k, v in data.items() if "MoE NSGAII" in k and "v2" not in k}
nsgaiiv2_data = {k: v for k, v in data.items() if "MoE NSGAIIv2" in k}

# Compute Pareto front for fixed method
fixed_vals = np.array(list(fixed_data.values()))

def pareto_frontier(points, maximize=False):
    points = points.copy()
    if maximize:
        points = -points
    sorted_idx = np.argsort(points[:, 0])
    points = points[sorted_idx]
    frontier = []
    min_second = np.inf
    for p in points:
        if p[1] < min_second:
            frontier.append(p)
            min_second = p[1]
    return np.array(frontier)

# ========== FIGURE 1: 2x2 Layout ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
ax1, ax2, ax3, ax4 = axes.flatten()

# Projection 1: Objective 1 vs Objective 2
ax1.scatter([v[0] for v in expert_data.values()], [v[1] for v in expert_data.values()],
            color='red', marker='s', s=100, label='Expert[0-2]', edgecolors='black', linewidth=1.5, zorder=5)
ax1.scatter([v[0] for v in fixed_data.values()], [v[1] for v in fixed_data.values()],
            color='blue', alpha=0.6, s=40, label='MoE Fixed', zorder=2)
ax1.scatter([v[0] for v in nsgaii_data.values()], [v[1] for v in nsgaii_data.values()],
            color='orange', alpha=0.5, s=30, label='NSGA-II', zorder=1)
ax1.scatter([v[0] for v in nsgaiiv2_data.values()], [v[1] for v in nsgaiiv2_data.values()],
            color='green', alpha=0.5, s=30, label='NSGA-IIv2', zorder=1)

fixed_frontier_12 = pareto_frontier(fixed_vals[:, :2])
ax1.plot(fixed_frontier_12[:, 0], fixed_frontier_12[:, 1], 'b-', linewidth=2, alpha=0.7, label='Fixed PF')

ax1.set_xlabel('Objective 1', fontsize=12)
ax1.set_ylabel('Objective 2', fontsize=12)
ax1.set_title('Projection: Obj1 vs Obj2', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=9)

# Projection 2: Objective 1 vs Objective 3
ax2.scatter([v[0] for v in expert_data.values()], [v[2] for v in expert_data.values()],
            color='red', marker='s', s=100, label='Expert[0-2]', edgecolors='black', linewidth=1.5, zorder=5)
ax2.scatter([v[0] for v in fixed_data.values()], [v[2] for v in fixed_data.values()],
            color='blue', alpha=0.6, s=40, label='MoE Fixed', zorder=2)
ax2.scatter([v[0] for v in nsgaii_data.values()], [v[2] for v in nsgaii_data.values()],
            color='orange', alpha=0.5, s=30, label='NSGA-II', zorder=1)
ax2.scatter([v[0] for v in nsgaiiv2_data.values()], [v[2] for v in nsgaiiv2_data.values()],
            color='green', alpha=0.5, s=30, label='NSGA-IIv2', zorder=1)

fixed_frontier_13 = pareto_frontier(np.column_stack([fixed_vals[:, 0], fixed_vals[:, 2]]))
ax2.plot(fixed_frontier_13[:, 0], fixed_frontier_13[:, 1], 'b-', linewidth=2, alpha=0.7)

ax2.set_xlabel('Objective 1', fontsize=12)
ax2.set_ylabel('Objective 3', fontsize=12)
ax2.set_title('Projection: Obj1 vs Obj3', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=9)

# Projection 3: Objective 2 vs Objective 3
ax3.scatter([v[1] for v in expert_data.values()], [v[2] for v in expert_data.values()],
            color='red', marker='s', s=100, label='Expert[0-2]', edgecolors='black', linewidth=1.5, zorder=5)
ax3.scatter([v[1] for v in fixed_data.values()], [v[2] for v in fixed_data.values()],
            color='blue', alpha=0.6, s=40, label='MoE Fixed', zorder=2)
ax3.scatter([v[1] for v in nsgaii_data.values()], [v[2] for v in nsgaii_data.values()],
            color='orange', alpha=0.5, s=30, label='NSGA-II', zorder=1)
ax3.scatter([v[1] for v in nsgaiiv2_data.values()], [v[2] for v in nsgaiiv2_data.values()],
            color='green', alpha=0.5, s=30, label='NSGA-IIv2', zorder=1)

fixed_frontier_23 = pareto_frontier(np.column_stack([fixed_vals[:, 1], fixed_vals[:, 2]]))
ax3.plot(fixed_frontier_23[:, 0], fixed_frontier_23[:, 1], 'b-', linewidth=2, alpha=0.7)

ax3.set_xlabel('Objective 2', fontsize=12)
ax3.set_ylabel('Objective 3', fontsize=12)
ax3.set_title('Projection: Obj2 vs Obj3', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best', fontsize=9)

# Legend subplot
ax4.axis('off')
legend_text = """
Pareto Front Summary:

• Fixed MoE configurations show clear trade-off patterns
• NSGA-II and NSGA-IIv2 find diverse solutions
• Experts serve as reference points
• Lower values are better for all objectives
"""
ax4.text(0.1, 0.5, legend_text, fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))

plt.suptitle('Multi-Objective Optimization Results - 2D Projections', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save Figure 1
fig1_path = os.path.join(output_dir, 'pareto_2x2_projections.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig1_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {fig1_path}")
print(f"✓ Saved: {fig1_path.replace('.png', '.pdf')}")

# ========== FIGURE 2: Three Separate High-Quality Plots ==========
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

projections = [
    (0, 1, 'Objective 1', 'Objective 2', 'Obj1_vs_Obj2'),
    (0, 2, 'Objective 1', 'Objective 3', 'Obj1_vs_Obj3'),
    (1, 2, 'Objective 2', 'Objective 3', 'Obj2_vs_Obj3')
]

for idx, (ax, (x_idx, y_idx, xlabel, ylabel, filename)) in enumerate(zip(axes2, projections)):
    ax.scatter([v[x_idx] for v in expert_data.values()], [v[y_idx] for v in expert_data.values()],
               color='red', marker='s', s=120, label='Experts', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter([v[x_idx] for v in fixed_data.values()], [v[y_idx] for v in fixed_data.values()],
               color='blue', alpha=0.5, s=50, label='Fixed MoE', zorder=2)
    ax.scatter([v[x_idx] for v in nsgaii_data.values()], [v[y_idx] for v in nsgaii_data.values()],
               color='orange', alpha=0.5, s=40, label='NSGA-II', zorder=1)
    ax.scatter([v[x_idx] for v in nsgaiiv2_data.values()], [v[y_idx] for v in nsgaiiv2_data.values()],
               color='green', alpha=0.5, s=40, label='NSGA-IIv2', zorder=1)
    
    frontier = pareto_frontier(np.column_stack([fixed_vals[:, x_idx], fixed_vals[:, y_idx]]))
    ax.plot(frontier[:, 0], frontier[:, 1], 'b-', linewidth=2.5, alpha=0.8, label='Fixed PF')
    
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f'{xlabel} vs {ylabel}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)

plt.suptitle('MoE Pareto Front Analysis - 2D Projections', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save Figure 2
fig2_path = os.path.join(output_dir, 'pareto_3panel_projections.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig2_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {fig2_path}")
print(f"✓ Saved: {fig2_path.replace('.png', '.pdf')}")

# ========== FIGURE 3: Individual High-Resolution Plots ==========
print("\nGenerating individual high-resolution plots...")

for idx, (x_idx, y_idx, xlabel, ylabel, filename) in enumerate(projections, 1):
    fig_ind, ax_ind = plt.subplots(1, 1, figsize=(10, 8))
    
    ax_ind.scatter([v[x_idx] for v in expert_data.values()], [v[y_idx] for v in expert_data.values()],
                   color='red', marker='s', s=150, label='Experts', edgecolors='black', linewidth=2, zorder=5)
    ax_ind.scatter([v[x_idx] for v in fixed_data.values()], [v[y_idx] for v in fixed_data.values()],
                   color='blue', alpha=0.6, s=80, label='Fixed MoE', zorder=2)
    ax_ind.scatter([v[x_idx] for v in nsgaii_data.values()], [v[y_idx] for v in nsgaii_data.values()],
                   color='orange', alpha=0.5, s=60, label='NSGA-II', zorder=1)
    ax_ind.scatter([v[x_idx] for v in nsgaiiv2_data.values()], [v[y_idx] for v in nsgaiiv2_data.values()],
                   color='green', alpha=0.5, s=60, label='NSGA-IIv2', zorder=1)
    
    frontier = pareto_frontier(np.column_stack([fixed_vals[:, x_idx], fixed_vals[:, y_idx]]))
    ax_ind.plot(frontier[:, 0], frontier[:, 1], 'b-', linewidth=3, alpha=0.8, label='Fixed MoE Pareto Front')
    
    ax_ind.set_xlabel(xlabel, fontsize=14)
    ax_ind.set_ylabel(ylabel, fontsize=14)
    ax_ind.set_title(f'{xlabel} vs {ylabel} - Pareto Front Analysis', fontsize=16, fontweight='bold')
    ax_ind.grid(True, alpha=0.3, linestyle='--')
    ax_ind.legend(loc='best', fontsize=12)
    
    # Add statistics box
    stats_text = f'Fixed MoE Points: {len(fixed_data)}\nPareto Front Size: {len(frontier)}'
    ax_ind.text(0.02, 0.98, stats_text, transform=ax_ind.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save individual plot
    ind_path = os.path.join(output_dir, f'pareto_{filename}.png')
    plt.savefig(ind_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(ind_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {ind_path}")
    print(f"✓ Saved: {ind_path.replace('.png', '.pdf')}")
    plt.close(fig_ind)

# ========== FIGURE 4: 3D Pareto Front ==========
from mpl_toolkits.mplot3d import Axes3D

fig3d = plt.figure(figsize=(12, 9))
ax3d = fig3d.add_subplot(111, projection='3d')

# Plot all points
ax3d.scatter([v[0] for v in fixed_data.values()], [v[1] for v in fixed_data.values()], [v[2] for v in fixed_data.values()],
             color='blue', alpha=0.5, s=40, label='Fixed MoE')
ax3d.scatter([v[0] for v in nsgaii_data.values()], [v[1] for v in nsgaii_data.values()], [v[2] for v in nsgaii_data.values()],
             color='orange', alpha=0.4, s=30, label='NSGA-II')
ax3d.scatter([v[0] for v in nsgaiiv2_data.values()], [v[1] for v in nsgaiiv2_data.values()], [v[2] for v in nsgaiiv2_data.values()],
             color='green', alpha=0.4, s=30, label='NSGA-IIv2')
ax3d.scatter([v[0] for v in expert_data.values()], [v[1] for v in expert_data.values()], [v[2] for v in expert_data.values()],
             color='red', marker='s', s=100, label='Experts', edgecolors='black', linewidth=1.5)

ax3d.set_xlabel('Objective 1', fontsize=12)
ax3d.set_ylabel('Objective 2', fontsize=12)
ax3d.set_zlabel('Objective 3', fontsize=12)
ax3d.set_title('3D Pareto Front - All Methods', fontsize=16, fontweight='bold')
ax3d.legend(loc='best', fontsize=10)

# Save 3D plot
fig3d_path = os.path.join(output_dir, 'pareto_3d_view.png')
plt.savefig(fig3d_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig3d_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {fig3d_path}")
print(f"✓ Saved: {fig3d_path.replace('.png', '.pdf')}")

# ========== SAVE DATA AND STATISTICS ==========
print("\nSaving statistics and data...")

# Save Pareto frontier statistics to text file
stats_path = os.path.join(output_dir, 'pareto_statistics.txt')
with open(stats_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("PARETO FRONT ANALYSIS STATISTICS\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Total fixed configurations: {len(fixed_data)}\n")
    f.write(f"Pareto front size (Obj1-Obj2): {len(fixed_frontier_12)}\n")
    f.write(f"Pareto front size (Obj1-Obj3): {len(fixed_frontier_13)}\n")
    f.write(f"Pareto front size (Obj2-Obj3): {len(fixed_frontier_23)}\n\n")
    
    f.write("Pareto-optimal fixed configurations (Obj1 vs Obj2):\n")
    for p in fixed_frontier_12:
        f.write(f"  ({p[0]:.4f}, {p[1]:.4f})\n")
    
    f.write("\nPareto-optimal fixed configurations (Obj1 vs Obj3):\n")
    for p in fixed_frontier_13:
        f.write(f"  ({p[0]:.4f}, {p[1]:.4f})\n")
    
    f.write("\nPareto-optimal fixed configurations (Obj2 vs Obj3):\n")
    for p in fixed_frontier_23:
        f.write(f"  ({p[0]:.4f}, {p[1]:.4f})\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("METHOD SUMMARY\n")
    f.write("=" * 80 + "\n")
    f.write(f"Experts: {len(expert_data)}\n")
    f.write(f"Fixed MoE: {len(fixed_data)}\n")
    f.write(f"NSGA-II: {len(nsgaii_data)}\n")
    f.write(f"NSGA-IIv2: {len(nsgaiiv2_data)}\n")

print(f"✓ Saved statistics: {stats_path}")

# Save raw data to CSV
import csv
csv_path = os.path.join(output_dir, 'pareto_data.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Method', 'Objective1', 'Objective2', 'Objective3'])
    for name, vals in data.items():
        writer.writerow([name, vals[0], vals[1], vals[2]])
print(f"✓ Saved raw data: {csv_path}")

print("\n" + "=" * 60)
print(f"✅ ALL FIGURES AND DATA SAVED TO: {output_dir}/")
print("=" * 60)
print("\nGenerated files:")
print("  📊 Figures:")
print("     - pareto_2x2_projections.png/pdf")
print("     - pareto_3panel_projections.png/pdf")
print("     - pareto_Obj1_vs_Obj2.png/pdf")
print("     - pareto_Obj1_vs_Obj3.png/pdf")
print("     - pareto_Obj2_vs_Obj3.png/pdf")
print("     - pareto_3d_view.png/pdf")
print("  📄 Data:")
print("     - pareto_statistics.txt")
print("     - pareto_data.csv")

# Show all plots
plt.show()