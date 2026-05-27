import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))

# Transparent background
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# ------------------------------------------------------------
# 1. Dashed gray frontier line
# ------------------------------------------------------------
ax.plot(
    [0.15, 0.75],
    [0.85, 0.25],
    linestyle=(0, (6, 5)),
    color="#B0B0B0",
    linewidth=5,
    zorder=1
)

# ------------------------------------------------------------
# 2. Hollow gray circle
# ------------------------------------------------------------
ax.scatter(
    [0.28],
    [0.72],
    s=650,
    facecolors="white",
    edgecolors="#7A7A7A",
    linewidths=4,
    zorder=3
)

# ------------------------------------------------------------
# 3. Blue selected star
# ------------------------------------------------------------
ax.scatter(
    [0.72],
    [0.25],
    s=3000,
    marker="*",
    facecolors="royalblue",
    edgecolors="navy",
    linewidths=4,
    zorder=4
)

# Formatting
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout(pad=0)

plt.savefig(
    "./plots/two_elements.svg",
    bbox_inches="tight",
    transparent=True
)

plt.show()