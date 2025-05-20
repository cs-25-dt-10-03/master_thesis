import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from flexoffer_logic import DFO, DependencyPolygon, Point

def plot_dfo(dfo):
    polygons = dfo.polygons
    num_polygons = len(polygons)

    fig, axes = plt.subplots(1, num_polygons, figsize=(num_polygons * 5, 5), constrained_layout=True)
    if num_polygons == 1:
        axes = [axes]

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, (polygon, ax) in enumerate(zip(polygons, axes)):
        x_vals = np.array([p.x for p in polygon.points])
        y_vals = np.array([p.y for p in polygon.points])

        color = colors[i % len(colors)]

        if len(polygon.points) == 2:
            ax.plot(x_vals, y_vals, linestyle='-', marker='o', color=color, alpha=0.6)
        elif len(polygon.points) >= 3:
            points_array = np.column_stack((x_vals, y_vals))
            hull = ConvexHull(points_array)
            hull_vertices = hull.vertices
            hull_vertices = np.append(hull_vertices, hull_vertices[0])
            ax.plot(x_vals[hull_vertices], y_vals[hull_vertices], linestyle='-', marker='o', color=color, alpha=0.6)
            ax.fill(x_vals[hull_vertices], y_vals[hull_vertices], color=color, alpha=0.2)

        ax.scatter(x_vals, y_vals, color=color, s=40, label=f"Polygon {i}")
        ax.set_xlabel("Previous Energy")
        ax.set_ylabel("Current Energy")
        ax.set_title(f"Polygon {i}")
        ax.grid(True)
        ax.legend()

    fig.suptitle(f"DFO ID: {dfo.dfo_id} - Dependency Polygons", fontsize=14)
    plt.show()