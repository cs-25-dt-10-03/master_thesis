from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from typing import List, Optional

class Point:
    def __init__(self, x: float, y: float):
        self.x = x  # Total energy from previous timesteps
        self.y = y  # Energy usage in the current timestep; either min or max

    def __repr__(self):
        return f"({self.x:.3g}, {self.y:.3g})"  # Three significant digits

class DependencyPolygon:
    def __init__(self, min_prev: float, max_prev: float, numsamples: int):
        self.points: List[Point] = []
        self.min_prev_energy = min_prev
        self.max_prev_energy = max_prev
        self.numsamples = numsamples

    def generate_polygon(self, charging_power: float, next_min_prev: float, next_max_prev: float):
        if self.min_prev_energy == self.max_prev_energy:
            min_current_energy = max(next_min_prev - self.min_prev_energy, 0.0)
            min_current_energy = min(min_current_energy, charging_power) # Limit to charging power
            max_current_energy = max(next_max_prev - self.min_prev_energy, 0.0)
            max_current_energy = min(max_current_energy, charging_power) # Limit to charging power
            
            self.add_point(self.min_prev_energy, min_current_energy)
            self.add_point(self.max_prev_energy, max_current_energy)
            return
        
        step = (self.max_prev_energy - self.min_prev_energy) / (self.numsamples - 1)

        for i in range(self.numsamples):
            current_prev_energy = self.min_prev_energy + i * step

            # Calculate the min and max energy needed for the next time slice
            min_current_energy = max(next_min_prev - current_prev_energy, 0.0)
            min_current_energy = min(min_current_energy, charging_power) # Limit to charging power
            max_current_energy = max(next_max_prev - current_prev_energy, 0.0)
            max_current_energy = min(max_current_energy, charging_power) # Limit to charging power

            # Add the points to the polygon
            self.add_point(current_prev_energy, min_current_energy)
            self.add_point(current_prev_energy, max_current_energy)

        self.sort_points()

    def generate_last_polygon(self):
        # Placeholder for last polygon logic
        pass

    def add_point(self, x: float, y: float):
        self.points.append(Point(x, y))

    def sort_points(self):
        self.points.sort(key=lambda p: (p.x, p.y))

    def print_polygon(self, i: int):
        print(f"Polygon {i}:")
        for point in self.points:
            print(point)

    def __repr__(self): #so print(polygon) can be used directly
        points_str = "\n".join(map(str, self.points))
        return f"Polygon:\n{points_str}"

class DFO:
    def __init__(
        self, 
        dfo_id: int, 
        min_prev: List[float], 
        max_prev: List[float], 
        numsamples: int = 5,
        charging_power: Optional[float] = None,
        min_total_energy: Optional[float] = None, 
        max_total_energy: Optional[float] = None,
        earliest_start: Optional[datetime] = None
    ):
        self.dfo_id = dfo_id
        self.polygons: List[DependencyPolygon] = [
            DependencyPolygon(min_p, max_p, numsamples) for min_p, max_p in zip(min_prev, max_prev)
        ]

        self.charging_power = charging_power if charging_power is not None else 7.3  # Default charging power
        self.min_total_energy = min_total_energy if min_total_energy is not None else min_prev[-1] # Last element in min_prev
        self.max_total_energy = max_total_energy if max_total_energy is not None else max_prev[-1] # Last element in max_prev

        self.earliest_start = earliest_start if earliest_start is not None else datetime.now()
        self.latest_start = self.earliest_start
     
    def generate_dependency_polygons(self):
        for i in range(len(self.polygons)):
            if i < len(self.polygons) - 1: # Generate allowed energy usage based on min/max dependency from the next timestep
                self.polygons[i].generate_polygon(self.charging_power, self.polygons[i + 1].min_prev_energy, self.polygons[i + 1].max_prev_energy)
        self.polygons = self.polygons[:-1]  # Remove the last polygon, as it was only there such that the loop could generate the second-to-last polygon
    
    def print_dfo(self):
        print(f"DFO ID: {self.dfo_id}")
        for i, polygon in enumerate(self.polygons):
            polygon.print_polygon(i)
    
    def __repr__(self): # so print(dfo) can be used directly
        polygons_str = "\n".join(
            f"Polygon {i}:\n{polygon}" for i, polygon in enumerate(self.polygons)
        )
        return f"DFO ID: {self.dfo_id}\n{polygons_str}"
    
    def plot_dfo(self):
        """Visualizes each dependency polygon in its own coordinate system."""
        num_polygons = len(self.polygons)
        fig, axes = plt.subplots(1, num_polygons, figsize=(num_polygons * 5, 5), constrained_layout=True)

        if num_polygons == 1:
            axes = [axes]  # Ensure axes is iterable when there's only one polygon

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Cycle through colors

        for i, (polygon, ax) in enumerate(zip(self.polygons, axes)):
            x_vals = np.array([point.x for point in polygon.points])
            y_vals = np.array([point.y for point in polygon.points])

            color = colors[i % len(colors)]  # Cycle colors if there are many polygons

            if len(polygon.points) == 2:  # If only two points, draw a single line
                ax.plot(x_vals, y_vals, linestyle='-', marker='o', color=color, alpha=0.6)
            elif len(polygon.points) >= 3:  # Convex Hull requires at least 3 points
                points_array = np.column_stack((x_vals, y_vals))
                hull = ConvexHull(points_array)
                hull_vertices = hull.vertices

                # Close the polygon by appending the first point at the end
                hull_vertices = np.append(hull_vertices, hull_vertices[0])

                ax.plot(x_vals[hull_vertices], y_vals[hull_vertices], linestyle='-', marker='o', color=color, alpha=0.6)
                ax.fill(x_vals[hull_vertices], y_vals[hull_vertices], color=color, alpha=0.2)  # Light fill

            ax.scatter(x_vals, y_vals, color=color, s=40, label=f"Polygon {i}")

            ax.set_xlabel("Previous Energy")
            ax.set_ylabel("Current Energy")
            ax.set_title(f"Polygon {i}")
            ax.legend()
            ax.grid(True)

        fig.suptitle(f"DFO ID: {self.dfo_id} - Dependency Polygons", fontsize=14)
        plt.show()
