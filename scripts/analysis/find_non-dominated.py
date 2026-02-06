import numpy as np

data = """
-0.0287 -0.5534
-0.0326 -0.5520
-0.0328 -0.5500
-0.0190 -0.5607
-0.0426 -0.5520
-0.0470 -0.5524 
-0.0253 -0.5538 
-0.0281 -0.5535
-0.0314 -0.5533
-0.0353 -0.5521
"""

# Parse data
points = []
for line in data.strip().split('\n'):
    parts = line.split()  # split() handles any whitespace
    x, y = float(parts[0]), float(parts[1])
    points.append((x, y))

points = np.array(points)

# Find non-dominated solutions (higher is better)
def is_dominated(point, other_points):
    for other in other_points:
        if np.all(other >= point) and np.any(other > point):
            return True
    return False

non_dominated = []
non_dominated_idx = []
for i, point in enumerate(points):
    others = np.delete(points, i, axis=0)
    if not is_dominated(point, others):
        non_dominated.append(point)
        non_dominated_idx.append(i)

non_dominated = np.array(non_dominated)

# Sort by first objective
sorted_idx = np.argsort(non_dominated[:, 0])[::-1]
non_dominated_sorted = non_dominated[sorted_idx]

print("Non-dominated (Pareto optimal) solutions:")
print("Obj1\tObj2")
for p in non_dominated_sorted:
    print(f"{p[0]:.4f}\t{p[1]:.4f}")

print(f"\nTotal: {len(non_dominated)} non-dominated solutions out of {len(points)}")