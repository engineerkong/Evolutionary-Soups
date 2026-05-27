import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))

# ============================================================
# Axes (thicker and larger arrows)
# ============================================================
ax.annotate(
    "",
    xy=(1.20, 0),
    xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=4.5,              # thicker axis
        color="black",
        mutation_scale=32    # larger arrow head
    )
)

ax.annotate(
    "",
    xy=(0, 1.20),
    xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=4.5,
        color="black",
        mutation_scale=32
    )
)

# ============================================================
# Dashed omega vector
# ============================================================
ann = ax.annotate(
    "",
    xy=(0.78, 0.78),
    xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=4.0,                 # thicker line
        linestyle="--",
        color="royalblue",
        mutation_scale=30       # larger arrow head
    )
)

# ------------------------------------------------------------
# Increase dash spacing
# ------------------------------------------------------------
ann.arrow_patch.set_linestyle((0, (8, 6)))
# format: (offset, (dash_length, gap_length))

# ============================================================
# Larger omega text
# ============================================================
ax.text(
    0.82,
    0.80,
    r'$\mathbf{\mu}$',
    fontsize=32,               # larger text
    color="midnightblue"
)

# ============================================================
# Formatting
# ============================================================
ax.set_xlim(-0.05, 1.30)
ax.set_ylim(-0.05, 1.30)

ax.set_aspect("equal")

ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

# transparent background
fig.patch.set_alpha(0)
ax.set_facecolor('none')

plt.tight_layout()

plt.savefig(
    "./plots/vector_w.png",
    dpi=300,
    bbox_inches='tight',
    transparent=True
)

plt.savefig(
    "./plots/vector_w.svg",
    dpi=300,
    bbox_inches='tight',
    transparent=True
)

plt.show()