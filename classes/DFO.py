from typing import List

class Point:
    def __init__(self, x: float, y: float):
        self.x = x  # Total energy from previous timesteps
        self.y = y  # Energy usage in the current timestep; either min or max

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

class DependencyPolygon:
    def __init__(self, min_prev: float, max_prev: float, numsamples: int):
        self.points: List[Point] = []
        self.min_prev_energy = min_prev
        self.max_prev_energy = max_prev
        self.numsamples = numsamples

    def generate_polygon(self, i: int, next_min_prev: float, next_max_prev: float):
        if self.min_prev_energy == self.max_prev_energy:
            min_current_energy = max(next_min_prev - self.min_prev_energy, 0.0)
            max_current_energy = max(next_max_prev - self.min_prev_energy, 0.0)
            
            self.add_point(self.min_prev_energy, min_current_energy)
            self.add_point(self.max_prev_energy, max_current_energy)
            return
        
        step = (self.max_prev_energy - self.min_prev_energy) / (self.numsamples - 1)

        for i in range(self.numsamples):
            current_prev_energy = self.min_prev_energy + i * step

            # Calculate the min and max energy needed for the next time slice
            min_current_energy = max(next_min_prev - current_prev_energy, 0.0)
            max_current_energy = max(next_max_prev - current_prev_energy, 0.0)

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

class DFO:
    def __init__(self, dfo_id: int, min_prev: List[float], max_prev: List[float], numsamples: int):
        self.dfo_id = dfo_id
        self.polygons: List[DependencyPolygon] = [
            DependencyPolygon(min_p, max_p, numsamples) for min_p, max_p in zip(min_prev, max_prev)
        ]

    def generate_dependency_polygons(self):
        for i in range(len(self.polygons)):
            if i < len(self.polygons) - 1:
                self.polygons[i].generate_polygon(i, self.polygons[i + 1].min_prev_energy, self.polygons[i + 1].max_prev_energy)
            else:
                self.polygons[i].generate_polygon(i, 0, 0)  # Last timestep

    def print_dfo(self):
        print(f"DFO ID: {self.dfo_id}")
        for i, polygon in enumerate(self.polygons):
            polygon.print_polygon(i)